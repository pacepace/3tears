"""the shape of a single-key KV watch: what it yields, and the slice of a bucket that offers it.

Pure python, with no nats-py import, so the in-memory double in
:mod:`threetears.core.testing.kv` can yield the same type the real watch does without dragging the
NATS client into a process that only runs the L1 tier. The real watch is
:meth:`threetears.nats.kv.NatsKvBucket.watch_key`.

**Why a key watch is its own primitive.** nats-py's ``KeyValue.watch`` creates an UNNAMED consumer
(``$JS.API.CONSUMER.CREATE.{stream}``), whose filter rides only in the request body where it could
name the whole bucket. A grant narrowed to one key (``JsCapability.KV_KEY_READ``) therefore cannot
admit it, and an ungranted JetStream call does not raise: it blocks to its deadline, so a watcher
built on the stock watch looks exactly like an unreachable broker. ``watch_key`` creates a NAMED
consumer instead, whose filter rides in the create SUBJECT, where a subject grant can pin it to one
key and the server checks it against the body.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass
from datetime import timedelta
from typing import Final, Protocol, runtime_checkable

__all__ = [
    "DEFAULT_KEY_WATCH_HEARTBEAT",
    "DEFAULT_KEY_WATCH_RETRY",
    "KvKeyUpdate",
    "KvKeyWatching",
]

#: how often the server proves a key watch's consumer is alive when the key is quiet. Three missed
#: beats and the watch replaces the consumer, so this bounds how long a consumer the server lost --
#: a broker restart empties a memory-backed stream and every consumer on it -- goes unnoticed.
DEFAULT_KEY_WATCH_HEARTBEAT: Final[timedelta] = timedelta(seconds=5)

#: how long a key watch waits after a consumer create failed before trying again.
DEFAULT_KEY_WATCH_RETRY: Final[timedelta] = timedelta(seconds=5)


@dataclass(frozen=True, slots=True)
class KvKeyUpdate:
    """one message on a watched key: a value, or the marker a delete or purge leaves.

    :ivar key: the watched key
    :ivar value: the stored bytes, or ``None`` when this message is a delete or purge marker
    :ivar revision: the message's stream sequence -- the key's revision as ``get_latest`` reports it
    """

    key: str
    value: bytes | None
    revision: int

    @property
    def deleted(self) -> bool:
        """whether this message removed the key rather than set it.

        :return: ``True`` for a delete or purge marker
        :rtype: bool
        """
        return self.value is None


@runtime_checkable
class KvKeyWatching(Protocol):
    """a bucket that can watch one key -- the slice of :class:`~threetears.nats.kv.NatsKvBucket` a watcher needs.

    Separate from ``KvBucketLike`` on purpose: adding a required method there would un-satisfy every
    double in every consuming repo that implements the operations it advertises, to serve the few
    callers that watch.
    """

    def watch_key(
        self,
        *,
        key: str,
        heartbeat: timedelta = DEFAULT_KEY_WATCH_HEARTBEAT,
        retry: timedelta = DEFAULT_KEY_WATCH_RETRY,
    ) -> AsyncIterator[KvKeyUpdate]:
        """the key's latest message, then every later one, until the caller stops iterating.

        :param key: the key to watch
        :ptype key: str
        :param heartbeat: how often a quiet consumer proves it is alive
        :ptype heartbeat: timedelta
        :param retry: the pause after a consumer create failed
        :ptype retry: timedelta
        :return: the key's messages, in order
        :rtype: AsyncIterator[KvKeyUpdate]
        """
        ...
