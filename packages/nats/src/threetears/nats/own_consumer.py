"""a JetStream delivery is accepted only when it came from the receiver's OWN consumer.

A receiver that reads a push consumer's deliver inbox, or a pull consumer's fetch inbox, cannot
trust the inbox name: any client in the account can get bytes delivered onto a subject it can name,
through a push consumer of its own whose ``deliver_subject`` is that inbox, or as the reply to a
request whose reply subject is that inbox, or by a plain publish. Only a delivery the server made
for the receiver's consumer carries that consumer's ack reply, ``$JS.ACK.<stream>.<consumer>.…``
(or the domain form, ``$JS.ACK.<domain>.<account>.<stream>.<consumer>.…``). A forger's own consumer
names a different consumer, a plain publish carries no ack reply or one the forger made up -- and
the consumer's name is drawn from :func:`random_consumer_name`, so a made-up one cannot match.

**What it rests on.** The name is unguessable only while no other principal can read it. A grant
that lets a principal list or inspect the consumers on the receiver's stream
(``$JS.API.CONSUMER.INFO|LIST|NAMES.<stream>``), or subscribe to that stream's consumer advisories
(``$JS.EVENT.ADVISORY.CONSUMER.>``), hands it the name and the deliver inbox, and this check then
authenticates nothing against that principal. Pod grants in this package hold none of those
(:mod:`threetears.nats.subject_permissions`); a repo granting more must not declare ``own-consumer``.

**What it does not cover.** A status frame (an idle heartbeat, ``409 Consumer Deleted``) carries no
ack reply, so it cannot be traced to a consumer at all. The receivers count a heartbeat as proof of
life and replace their consumer on any other status; the barrier against a forged one is the inbox
name, drawn from the CSPRNG, and a replacement moves the receiver onto a fresh random inbox the
forger does not know. A refused data delivery is never proof of life.

:func:`delivered_by_own_consumer` is the check, and logs one refusal event naming the receiver,
the reason and the reply the delivery claimed, never the payload.
"""

from __future__ import annotations

import secrets
from enum import StrEnum
from typing import Final, Protocol

from threetears.observe import get_logger

__all__ = [
    "ForeignDeliveryReason",
    "delivered_by_own_consumer",
    "random_consumer_name",
]

log = get_logger(__name__)

#: random bytes in a consumer name: 128 bits, as unguessable as the inbox it delivers to.
_CONSUMER_NAME_BYTES: Final[int] = 16

#: tokens in an ack reply without a JetStream domain, and the fewest in one with a domain; the
#: counts nats-py's ``Msg.Metadata`` parser accepts.
_V1_ACK_TOKENS: Final[int] = 9
_V2_MIN_ACK_TOKENS: Final[int] = 11

#: how much of a claimed reply subject a refusal logs; enough to recognise, bounded for the log line.
_CLAIM_LOG_CHARS: Final[int] = 200


class ForeignDeliveryReason(StrEnum):
    """why a delivery was refused as not from the receiver's own consumer."""

    NO_JETSTREAM_REPLY = "no_jetstream_reply"
    OTHER_STREAM = "other_stream"
    OTHER_CONSUMER = "other_consumer"


class _Delivered(Protocol):
    """the part of a nats-py ``Msg`` the check reads."""

    @property
    def reply(self) -> str: ...

    @property
    def subject(self) -> str: ...


def random_consumer_name(prefix: str) -> str:
    """a consumer name no other client can guess: ``prefix`` then 32 random hex characters.

    :param prefix: what the name starts with, so an operator can tell whose consumer it is
    :ptype prefix: str
    :return: the name
    :rtype: str
    """
    return f"{prefix}{secrets.token_hex(_CONSUMER_NAME_BYTES)}"


def _foreign_reason(reply: str, *, stream: str, consumer: str) -> ForeignDeliveryReason | None:
    """why ``reply`` is not ``consumer``'s ack reply on ``stream``, or ``None`` when it is.

    Reads the two ack-reply forms nats-server sends and nats-py's ``Msg.metadata`` parses: nine
    tokens (``$JS.ACK.<stream>.<consumer>.<delivered>.<sseq>.<cseq>.<ts>.<pending>``), and the
    domain form of eleven or more (``$JS.ACK.<domain>.<account>.<stream>.<consumer>.…``).

    :param reply: the delivery's reply subject
    :ptype reply: str
    :param stream: the receiver's stream
    :ptype stream: str
    :param consumer: the receiver's consumer name
    :ptype consumer: str
    :return: the reason, or ``None`` for the receiver's own delivery
    :rtype: ForeignDeliveryReason | None
    """
    tokens = reply.split(".")
    if tokens[:2] != ["$JS", "ACK"]:
        return ForeignDeliveryReason.NO_JETSTREAM_REPLY
    if len(tokens) == _V1_ACK_TOKENS:
        claimed_stream, claimed_consumer = tokens[2], tokens[3]
    elif len(tokens) >= _V2_MIN_ACK_TOKENS:
        claimed_stream, claimed_consumer = tokens[4], tokens[5]
    else:
        return ForeignDeliveryReason.NO_JETSTREAM_REPLY
    if claimed_stream != stream:
        return ForeignDeliveryReason.OTHER_STREAM
    if claimed_consumer != consumer:
        return ForeignDeliveryReason.OTHER_CONSUMER
    return None


def delivered_by_own_consumer(msg: _Delivered, *, stream: str, consumer: str, receiver: str) -> bool:
    """whether ``msg`` is a delivery the server made for ``consumer`` on ``stream``.

    A delivery that is not is refused: the caller drops it without acking it and keeps waiting.
    One warning event names the receiver, the reason and the reply the delivery claimed.

    :param msg: the delivered message
    :ptype msg: _Delivered
    :param stream: the stream the receiver's consumer reads
    :ptype stream: str
    :param consumer: the receiver's consumer name
    :ptype consumer: str
    :param receiver: what the receiver is, for the refusal event (``result waiter``, ``key watch``)
    :ptype receiver: str
    :return: ``True`` for the receiver's own delivery
    :rtype: bool
    """
    reply = msg.reply or ""
    reason = _foreign_reason(reply, stream=stream, consumer=consumer)
    if reason is None:
        return True
    log.warning(
        "%s refused a delivery not made by its own consumer (%s)",
        receiver,
        reason.value,
        extra={
            "extra_data": {
                "door": receiver,
                "reason": reason.value,
                "claimed_reply": reply[:_CLAIM_LOG_CHARS],
                "delivered_on": msg.subject,
                "stream": stream,
                "consumer": consumer,
            }
        },
    )
    return False
