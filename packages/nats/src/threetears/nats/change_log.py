"""what a change log holds, in terms free of the NATS client: its state, and one entry.

A change log (:class:`threetears.nats.kv.NatsChangeLog`) is an append-only log of keyed records over
one JetStream stream, whose sequence numbers each record. These are the values it answers with, here
so that a test double and a reader that never opens a connection can name them without importing
nats-py, and :class:`ChangeLogLike`, what a caller uses of the log (a client binds one by name through
:class:`~threetears.nats.capabilities.ChangeLogBinder`).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from typing import TYPE_CHECKING, Final, Protocol, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

    from threetears.nats.kv_watch import KvKeyUpdate

__all__ = [
    "CHANGE_LOG_CLEANUP_FLOOR_METADATA",
    "CHANGE_LOG_CONTINUES_METADATA",
    "CHANGE_LOG_COVERED_FROM_METADATA",
    "CHANGE_LOG_INCARNATION_METADATA",
    "CHANGE_LOG_RESTORING_METADATA",
    "ChangeLogEntry",
    "ChangeLogLike",
    "ChangeLogState",
    "cleanup_floor_metadata",
    "published_cleanup_floor",
]

#: the metadata key a change log's stream carries its incarnation under
CHANGE_LOG_INCARNATION_METADATA: Final = "threetears_incarnation"

#: the metadata keys a restored log names the incarnation it continues under, and the sequence of that
#: incarnation from which the restore's change notes cover every epoch (empty when they cover none): a
#: follower whose mark is at or past it walks every restored record and has missed nothing
CHANGE_LOG_CONTINUES_METADATA: Final = "threetears_continues"
CHANGE_LOG_COVERED_FROM_METADATA: Final = "threetears_covered_from"

#: the metadata key a log being restored names the lost incarnation under, from its declaration until the
#: restore's last append: the continuation (:data:`CHANGE_LOG_CONTINUES_METADATA`) is written only then, so a
#: follower never carries its mark over to a log that does not yet hold everything the notes name
CHANGE_LOG_RESTORING_METADATA: Final = "threetears_restoring"

#: the metadata key the log's owner (the hub's keeper) publishes the restore window's cleanup floor F under, as
#: ``<incarnation>:<F>``: what a reader that may not read the log by sequence (a pod) learns F from, in the same
#: ``STREAM.INFO`` it already reads the log's state with. A value naming another incarnation is not this log's floor,
#: and a log whose owner publishes none (an older hub) has none: that reader's quiet-time cleanup then removes nothing
CHANGE_LOG_CLEANUP_FLOOR_METADATA: Final = "threetears_cleanup_floor"


def published_cleanup_floor(value: str | None, incarnation: str) -> int | None:
    """the floor a log's metadata publishes (:data:`CHANGE_LOG_CLEANUP_FLOOR_METADATA`) for ``incarnation``.

    :param value: the metadata's value, or ``None`` when absent
    :ptype value: str | None
    :param incarnation: the log's incarnation
    :ptype incarnation: str
    :return: F, or ``None`` when absent, malformed, or published for another incarnation
    :rtype: int | None
    """
    named, _, floor = (value or "").rpartition(":")
    return int(floor) if named == incarnation and floor.isdigit() else None


def cleanup_floor_metadata(floor: int, incarnation: str) -> str:
    """the metadata value publishing ``floor`` for ``incarnation`` (:data:`CHANGE_LOG_CLEANUP_FLOOR_METADATA`).

    :param floor: F
    :ptype floor: int
    :param incarnation: the log's incarnation
    :ptype incarnation: str
    :return: ``<incarnation>:<F>``
    :rtype: str
    """
    return f"{incarnation}:{floor}"


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
    :ivar covered_from: the sequence of :attr:`continues` from which the restore's change notes cover every
        epoch, so a follower whose mark is at or past it misses nothing by walking the restored log;
        ``None`` when the notes cover none of it, or the log continues none
    :ivar max_age: how long the log keeps a record, as declared; ``None`` when unbounded or not said
    :ivar restoring: for a log whose restore has not finished, the incarnation of the lost log it is being
        restored from; ``None`` otherwise. Such a log continues nothing yet
    :ivar cleanup_floor: the restore window's cleanup floor F as the log's owner last published it for this
        incarnation (:data:`CHANGE_LOG_CLEANUP_FLOOR_METADATA`), at most one keeper mark old; ``None`` when it
        published none
    """

    first_seq: int
    last_seq: int
    bytes: int
    max_bytes: int | None
    incarnation: str
    continues: str | None = None
    covered_from: int | None = None
    max_age: timedelta | None = None
    restoring: str | None = None
    cleanup_floor: int | None = None


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


@runtime_checkable
class ChangeLogLike(Protocol):
    """what a caller uses of a change log: :class:`threetears.nats.kv.NatsChangeLog`, or its test double
    (:class:`threetears.core.testing.kv.FakeChangeLog`), each satisfying it structurally."""

    async def declare(
        self,
        *,
        max_age: timedelta,
        max_bytes: int,
        replicas: int | None,
        incarnation: str | None = None,
        first_seq: int | None = None,
        continues: tuple[str, int | None] | None = None,
        restoring: str | None = None,
    ) -> ChangeLogState: ...

    async def state(self) -> ChangeLogState | None: ...

    async def label_restored(self, *, continues: tuple[str, int | None]) -> ChangeLogState: ...

    async def label_cleanup_floor(self, floor: int) -> ChangeLogState: ...

    async def append(self, key: str, value: bytes, *, msg_id: str) -> int: ...

    async def last(self, key: str) -> ChangeLogEntry | None: ...

    async def at(self, seq: int) -> ChangeLogEntry | None: ...

    async def next_after(self, key: str, seq: int) -> ChangeLogEntry | None: ...

    async def records_after(self, key: str, seq: int, upto: int) -> list[ChangeLogEntry]: ...

    def watch(self, key: str) -> AsyncGenerator[KvKeyUpdate]: ...
