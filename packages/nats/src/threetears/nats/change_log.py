"""what a change log holds, in terms free of the NATS client: its state, and one entry.

A change log (:class:`threetears.nats.kv.NatsChangeLog`) is an append-only log of keyed records over
one JetStream stream, whose sequence numbers each record. These are the values it answers with, here
so that a test double and a reader that never opens a connection can name them without importing
nats-py.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

__all__ = [
    "CHANGE_LOG_CONTINUES_METADATA",
    "CHANGE_LOG_CONTINUES_THROUGH_METADATA",
    "CHANGE_LOG_INCARNATION_METADATA",
    "ChangeLogEntry",
    "ChangeLogState",
]

#: the metadata key a change log's stream carries its incarnation under
CHANGE_LOG_INCARNATION_METADATA: Final = "threetears_incarnation"

#: the metadata keys a restored log names the incarnation it continues under, and the last sequence of
#: that incarnation it put back: a sequence at or below it numbers the same epoch in both
CHANGE_LOG_CONTINUES_METADATA: Final = "threetears_continues"
CHANGE_LOG_CONTINUES_THROUGH_METADATA: Final = "threetears_continues_through"


@dataclass(frozen=True, slots=True)
class ChangeLogState:
    """what a change log's stream holds, as ``STREAM.INFO`` reports it.

    :ivar first_seq: the oldest sequence still held
    :ivar last_seq: the newest sequence written (0 when nothing has been)
    :ivar bytes: the bytes held
    :ivar max_bytes: the byte bound, or ``None`` when unbounded
    :ivar incarnation: the stream's incarnation, minted when it was declared, so a sequence under one
        incarnation is never compared with one under another
    :ivar continues: for a restored log, the incarnation of the lost log it continues; ``None`` otherwise
    :ivar continues_through: the last sequence of :attr:`continues` the restore put back: up to it, a
        sequence numbers the same epoch under both incarnations; past it, the lost log's epochs are gone
    """

    first_seq: int
    last_seq: int
    bytes: int
    max_bytes: int | None
    incarnation: str
    continues: str | None = None
    continues_through: int | None = None


@dataclass(frozen=True, slots=True)
class ChangeLogEntry:
    """one message of a change log.

    :ivar seq: its stream sequence
    :ivar key: the key it was appended under
    :ivar value: its bytes
    """

    seq: int
    key: str
    value: bytes
