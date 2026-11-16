"""move a JetStream stream (or KV bucket) live on file storage onto memory storage.

**Why.** NATS is the L2 tier: every stream and bucket is memory storage on three servers, and
durability is those three copies plus L3 (owner's rule, months old; restated 2026-11-15). Streams
created before a declaration moved to memory stay on file storage: JetStream cannot change a live
stream's storage, and :meth:`threetears.nats.NatsClient.ensure_jetstream_stream` leaves such a
stream exactly as it is. This is the operator's move: create the stream again on memory, keeping
what it still owes.

**What is kept.**

- a KV bucket (``KV_<bucket>``): the latest value of every key. Delete and purge markers are not
  copied; the bucket's history before the latest value is not.
- a plain stream: every message no durable consumer reads -- the dead letters parked on a subject no
  consumer filters on -- not older than the stream's new ``max_age``. A message a consumer reads is
  never copied: the move is refused while any durable consumer still owes one (pending or
  unacknowledged), so everything a consumer reads has been handled already, and a copy would hand it
  to the consumer a second time.
- every durable consumer, created again with its own config, so the services bound to it keep
  fetching (a pull consumer) or receiving (a push consumer's deliver subject) without a restart.
  An ephemeral consumer (a KV watcher) is not: its client must watch again.

**What a move costs.** Between the delete and the create (milliseconds) a publish to the stream is
refused; a publisher that does not retry loses that message. Sequence numbers start again, so a KV
revision held across the move no longer matches. Run it when the stream is quiet, then restart the
services that hold KV revisions or watches on it.

A dry run (:func:`stream_move_plan`) reads and changes nothing but the stream's info and, for a KV
bucket or a stream with parked messages, a read of each message it would copy.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any, Final, Protocol

from threetears.observe import get_logger

__all__ = [
    "StreamMove",
    "StreamMoveRefused",
    "StreamMoving",
    "moves_wanted",
    "stream_move",
    "stream_move_plan",
]

log = get_logger(__name__)

#: a KV bucket's backing stream carries this prefix
_KV_PREFIX: Final = "KV_"

#: the KV header that marks a delete or a purge; such a message is not a value
_KV_OPERATION_HEADER: Final = "KV-Operation"

#: headers a copy drops: they describe the publish into the old stream, and an expectation would be
#: refused by the new one
_DROPPED_HEADER_PREFIXES: Final = ("Nats-Expected-", "Nats-Msg-Id")

#: the stream-get filter matching every subject a stream holds
_ANY_SUBJECT: Final = ">"

#: how far a plan's detail lines sit under its first
_INDENT: Final = "  "

#: JetStream's storage words. nats-py's ``StorageType`` is a ``str`` enum over exactly these, so a
#: config read back compares equal to them and one carrying the word goes over the wire the same.
#: Naming the words keeps this module off nats-py: every JetStream call it makes goes through the
#: context it is handed (``NatsClient.jetstream_context()``), the wrapper's escape hatch.
_FILE: Final = "file"
_MEMORY: Final = "memory"


class StreamMoveRefused(Exception):
    """the move would lose something a consumer still owes, or the stream is not on file storage."""


class StreamMoving(Protocol):
    """the JetStream calls a move makes (nats-py's ``JetStreamContext``)."""

    async def stream_info(self, name: str) -> Any:
        """the live info of one stream."""
        ...

    async def consumers_info(self, stream: str, offset: int | None = None) -> list[Any]:
        """every consumer of a stream."""
        ...

    async def get_msg(
        self,
        stream_name: str,
        seq: int | None = None,
        subject: str | None = None,
        direct: bool | None = False,
        next: bool | None = False,
    ) -> Any:
        """one stored message by sequence."""
        ...

    async def delete_stream(self, name: str) -> bool:
        """delete a stream with every message and consumer in it."""
        ...

    async def add_stream(self, config: Any = None, **params: Any) -> Any:
        """create a stream."""
        ...

    async def add_consumer(self, stream: str, config: Any = None, **params: Any) -> Any:
        """create a consumer."""
        ...

    async def publish(
        self,
        subject: str,
        payload: bytes = b"",
        timeout: float | None = None,
        stream: str | None = None,
        headers: dict[str, Any] | None = None,
    ) -> Any:
        """publish one message, acknowledged by the stream."""
        ...


@dataclass(frozen=True)
class _Copy:
    """one message the move writes into the new stream."""

    subject: str
    data: bytes
    headers: dict[str, str] | None


@dataclass(frozen=True)
class StreamMove:
    """what moving one stream onto memory does, or did.

    :ivar stream: the stream's name
    :ivar kv: whether it backs a KV bucket
    :ivar storage: its storage when the plan was made (``file``, ``memory``)
    :ivar messages: the messages it held
    :ivar bytes: the bytes it held
    :ivar replicas: the servers the new stream is created on
    :ivar max_age_seconds: the new stream's age bound, 0 for none
    :ivar max_bytes: the new stream's byte bound, -1 for none
    :ivar copied: the messages written into the new stream (planned, on a dry run)
    :ivar skipped: messages not copied, by why (``consumed``, ``older than max_age``, ``kv marker``)
    :ivar consumers: the durable consumers created again
    :ivar ephemeral_consumers: the ephemeral consumers (watchers) the move ends
    :ivar refused: why it cannot move now, or ``None``
    :ivar moved: whether this call moved it
    """

    stream: str
    kv: bool
    storage: str
    messages: int
    bytes: int
    replicas: int
    max_age_seconds: float
    max_bytes: int
    copied: int
    skipped: Mapping[str, int] = field(default_factory=dict)
    consumers: tuple[str, ...] = ()
    ephemeral_consumers: tuple[str, ...] = ()
    refused: str | None = None
    moved: bool = False

    def summary(self) -> str:
        """the move as lines an operator reads.

        :return: the lines
        :rtype: str
        """
        verdict = "moved" if self.moved else ("REFUSED: " + self.refused if self.refused else "would move")
        skipped = ", ".join(f"{why} {n}" for why, n in sorted(self.skipped.items())) or "none"
        rows = (
            ("now", f"{self.storage}, {self.messages} messages, {self.bytes} bytes"),
            (
                "onto",
                f"memory on {self.replicas} servers, max_age {self.max_age_seconds:g}s, max_bytes {self.max_bytes}",
            ),
            ("copied", f"{self.copied} ({'latest value of each key' if self.kv else 'messages no consumer reads'})"),
            ("not copied", skipped),
            ("consumers kept", ", ".join(self.consumers) or "none"),
            ("watchers ended", ", ".join(self.ephemeral_consumers) or "none"),
        )
        return "\n".join([f"{self.stream}: {verdict}", *(f"{_INDENT}{label:<18} {value}" for label, value in rows)])


def _storage_name(storage: Any) -> str:
    """a JetStream storage type as its word (``file``, ``memory``).

    :param storage: nats-py's ``StorageType``, or its string value
    :ptype storage: Any
    :return: the word
    :rtype: str
    """
    return str(getattr(storage, "value", storage))


def _filters(consumer: Any) -> tuple[str, ...]:
    """the subjects a consumer reads: its filters, or every subject when it has none.

    :param consumer: a ``ConsumerInfo``
    :ptype consumer: Any
    :return: the filters
    :rtype: tuple[str, ...]
    """
    config = consumer.config
    subjects = tuple(config.filter_subjects or ())
    if config.filter_subject:
        subjects += (config.filter_subject,)
    return subjects or (_ANY_SUBJECT,)


def _matches(subject: str, pattern: str) -> bool:
    """whether a NATS subject matches a filter (``*`` one token, ``>`` the rest).

    :param subject: the subject
    :ptype subject: str
    :param pattern: the filter
    :ptype pattern: str
    :return: whether it matches
    :rtype: bool
    """
    tokens, wanted = subject.split("."), pattern.split(".")
    for index, token in enumerate(wanted):
        if token == ">":
            return len(tokens) > index
        if index >= len(tokens) or (token != "*" and token != tokens[index]):
            return False
    return len(tokens) == len(wanted)


def _kept_headers(headers: Mapping[str, str] | None) -> dict[str, str] | None:
    """a stored message's headers without the ones that describe its publish into the old stream.

    :param headers: the stored headers
    :ptype headers: Mapping[str, str] | None
    :return: the headers the copy carries, ``None`` when none is left
    :rtype: dict[str, str] | None
    """
    kept = {k: v for k, v in (headers or {}).items() if not k.startswith(_DROPPED_HEADER_PREFIXES)}
    return kept or None


async def _plan(
    js: StreamMoving, stream: str, *, replicas: int, max_age_seconds: float | None, max_bytes: int | None, now: datetime
) -> tuple[StreamMove, list[_Copy], Any, list[Any]]:
    """read the stream, its consumers and every message it would copy.

    :param js: a JetStream context with stream management
    :ptype js: StreamMoving
    :param stream: the stream's full name
    :ptype stream: str
    :param replicas: the servers the new stream goes on
    :ptype replicas: int
    :param max_age_seconds: the new age bound, ``None`` to keep the live one
    :ptype max_age_seconds: float | None
    :param max_bytes: the new byte bound, ``None`` to keep the live one
    :ptype max_bytes: int | None
    :param now: the time messages are aged against
    :ptype now: datetime
    :return: the plan, the copies, the live config and the durable consumers
    :rtype: tuple[StreamMove, list[_Copy], Any, list[Any]]
    """
    info = await js.stream_info(stream)
    config, state = info.config, info.state
    kv = stream.startswith(_KV_PREFIX)
    consumers = await js.consumers_info(stream)
    durables = [c for c in consumers if c.config.durable_name]
    ephemerals = [c for c in consumers if not c.config.durable_name]
    age = float(max_age_seconds if max_age_seconds is not None else (config.max_age or 0))
    byte_bound = int(max_bytes if max_bytes is not None else (config.max_bytes if config.max_bytes is not None else -1))
    refused: str | None = None
    if _storage_name(config.storage) != _FILE:
        refused = f"it is on {_storage_name(config.storage)} storage already"
    owing = {c.name: (int(c.num_pending or 0), int(c.num_ack_pending or 0)) for c in durables}
    owing = {name: counts for name, counts in owing.items() if counts != (0, 0)}
    if refused is None and owing:
        refused = "durable consumers still owe messages (pending, unacknowledged): " + ", ".join(
            f"{name} {pending}/{unacked}" for name, (pending, unacked) in sorted(owing.items())
        )
    copies: list[_Copy] = []
    skipped: dict[str, int] = {}
    read_filters = [f for c in durables for f in _filters(c)]
    oldest = now - timedelta(seconds=age) if age > 0 else None
    sequence, last = int(state.first_seq), int(state.last_seq)
    while refused is None and int(state.messages) and sequence <= last:
        stored = await js.get_msg(stream, sequence, subject=_ANY_SUBJECT, next=True)
        if int(stored.seq) > last:
            break
        sequence = int(stored.seq) + 1
        headers = dict(stored.headers or {})
        why: str | None = None
        if kv and headers.get(_KV_OPERATION_HEADER) in ("DEL", "PURGE"):
            why = "kv marker"
        elif not kv and any(_matches(stored.subject, f) for f in read_filters):
            why = "consumed"
        elif not kv and oldest is not None and stored.time is not None and stored.time < oldest:
            why = "older than max_age"
        if why is None:
            copies.append(_Copy(stored.subject, stored.data or b"", _kept_headers(headers)))
        else:
            skipped[why] = skipped.get(why, 0) + 1
    move = StreamMove(
        stream=stream,
        kv=kv,
        storage=_storage_name(config.storage),
        messages=int(state.messages),
        bytes=int(state.bytes),
        replicas=max(int(config.num_replicas or 1), replicas),
        max_age_seconds=age,
        max_bytes=byte_bound,
        copied=len(copies),
        skipped=skipped,
        consumers=tuple(sorted(c.name for c in durables)),
        ephemeral_consumers=tuple(sorted(c.name for c in ephemerals)),
        refused=refused,
    )
    return move, copies, config, durables


async def stream_move_plan(
    js: StreamMoving,
    stream: str,
    *,
    replicas: int,
    max_age_seconds: float | None = None,
    max_bytes: int | None = None,
    now: datetime | None = None,
) -> StreamMove:
    """what moving ``stream`` onto memory would do. Changes nothing.

    :param js: a JetStream context with stream management
    :ptype js: StreamMoving
    :param stream: the stream's full name (``aibots-audit``, ``KV_tool_catalog``)
    :ptype stream: str
    :param replicas: the servers the new stream goes on (never fewer than it is live on)
    :ptype replicas: int
    :param max_age_seconds: the new stream's age bound; ``None`` keeps the live one
    :ptype max_age_seconds: float | None
    :param max_bytes: the new stream's byte bound; ``None`` keeps the live one
    :ptype max_bytes: int | None
    :param now: the time messages are aged against; ``None`` for now
    :ptype now: datetime | None
    :return: the plan; ``refused`` says why it cannot move now
    :rtype: StreamMove
    """
    move, _copies, _config, _durables = await _plan(
        js,
        stream,
        replicas=replicas,
        max_age_seconds=max_age_seconds,
        max_bytes=max_bytes,
        now=now or datetime.now(UTC),
    )
    return move


async def stream_move(
    js: StreamMoving,
    stream: str,
    *,
    replicas: int,
    max_age_seconds: float | None = None,
    max_bytes: int | None = None,
    now: datetime | None = None,
) -> StreamMove:
    """move ``stream`` onto memory: delete it, create it again on memory, copy back what it keeps.

    :param js: a JetStream context with stream management
    :ptype js: StreamMoving
    :param stream: the stream's full name
    :ptype stream: str
    :param replicas: the servers the new stream goes on (never fewer than it is live on)
    :ptype replicas: int
    :param max_age_seconds: the new stream's age bound; ``None`` keeps the live one
    :ptype max_age_seconds: float | None
    :param max_bytes: the new stream's byte bound; ``None`` keeps the live one
    :ptype max_bytes: int | None
    :param now: the time messages are aged against; ``None`` for now
    :ptype now: datetime | None
    :return: what was done
    :rtype: StreamMove
    :raises StreamMoveRefused: it is not on file storage, or a durable consumer still owes a message
    :raises Exception: a JetStream call failed; the summary logged says how far the move got
    """
    move, copies, config, durables = await _plan(
        js,
        stream,
        replicas=replicas,
        max_age_seconds=max_age_seconds,
        max_bytes=max_bytes,
        now=now or datetime.now(UTC),
    )
    if move.refused is not None:
        raise StreamMoveRefused(f"{stream}: {move.refused}")
    created = dataclasses.replace(
        config,
        storage=_MEMORY,
        num_replicas=move.replicas,
        max_age=move.max_age_seconds,
        max_bytes=move.max_bytes,
    )
    log.warning(
        "moving stream onto memory: deleting it", extra={"extra_data": {"stream": stream, "copies": len(copies)}}
    )
    await js.delete_stream(stream)
    await js.add_stream(created)
    for consumer in durables:
        await js.add_consumer(stream, consumer.config)
    for copy in copies:
        await js.publish(copy.subject, copy.data, stream=stream, headers=copy.headers)
    moved = dataclasses.replace(move, moved=True)
    log.warning("stream moved onto memory", extra={"extra_data": {"stream": stream, "copied": moved.copied}})
    return moved


def moves_wanted(streams: Sequence[Any]) -> list[str]:
    """the names of the streams in a listing that are on file storage.

    :param streams: ``StreamInfo`` values
    :ptype streams: Sequence[Any]
    :return: their names, sorted
    :rtype: list[str]
    """
    return sorted(s.config.name for s in streams if _storage_name(s.config.storage) == _FILE)
