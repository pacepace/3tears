"""collection write generations over the epoch bucket.

:class:`EpochGenerationSource` implements
:class:`threetears.core.collections.generation.GenerationSource`: a collection that caches absences
stamps every recorded absence with the table's generation, and a collection switched on to carry a
write generation advances it once per commit and stamps the token on that commit's row broadcasts.
It reads and writes the bucket, so it is for a principal granted both -- the hub, the gateway.

:class:`EpochGenerationReader` is the other half, for a pod that only FOLLOWS: it binds the bucket,
reads a table's generation or watches its key, and never writes. A pod holds a read on the bucket
and no write, so that it cannot fake an advance the fleet acts on.

**One value per table, holding both halves of the answer: ``"{incarnation}:{count}"``.** The count
moves on with every committed write. The incarnation is minted when the value is first created, so
a broker restart that empties the memory-backed bucket forces a new one on the next read. Keeping
both in one value is what makes a single read trustworthy: reading an incarnation and a counter as
two keys can straddle a wipe and yield an old incarnation beside a reset count, or a new
incarnation beside an old count -- and the second can later equal a genuine token, revalidating an
absence that writes had already superseded.
"""

from __future__ import annotations

import asyncio
import random
from collections.abc import AsyncGenerator
from contextlib import aclosing
from typing import TYPE_CHECKING, Any, Final, Protocol, runtime_checkable

from threetears.core.exceptions import GenerationUnavailableError
from threetears.nats.errors import KvError
from threetears.nats.subjects import Subjects
from threetears.observe import get_logger
from uuid_utils import uuid7

from threetears.epoch.client import _EPOCH_BUCKET, _key_for

if TYPE_CHECKING:
    from threetears.nats.kv import KvBucketLike, KvCapable

__all__ = ["EpochGenerationReader", "EpochGenerationSource", "generation_kv_key"]

log = get_logger(__name__)

#: matched to the coordination counters' budget: contention on one table's generation is a burst of
#: concurrent writes to that table, which is exactly when a small budget would give up.
_MAX_CAS_ATTEMPTS: Final = 30

#: full-jitter backoff bound between compare-and-swap retries, seconds.
_CAS_RETRY_BACKOFF_SECONDS: Final = 0.02


def generation_kv_key(table_name: str) -> str:
    """the key in the epoch bucket holding ``table_name``'s write generation.

    The one derivation the source, the reader and a grant for either all use: a principal that
    follows a table is granted a read of exactly this key.

    :param table_name: the collection's table
    :ptype table_name: str
    :return: the KV key
    :rtype: str
    """
    return _key_for(Subjects.collection_generation_epoch(table_name))


@runtime_checkable
class _KeyWatchable(Protocol):
    """a bucket whose single keys can be watched; the real bucket and the shipped in-memory one are."""

    def watch_key(self, *, key: str) -> AsyncGenerator[Any]: ...


def _parse(table_name: str, raw: bytes) -> tuple[str, int]:
    """split a stored generation into its incarnation and count.

    :param table_name: the table, for the error
    :ptype table_name: str
    :param raw: the stored value
    :ptype raw: bytes
    :return: ``(incarnation, count)``
    :rtype: tuple[str, int]
    :raises GenerationUnavailableError: when the value is not ``incarnation:count``
    """
    incarnation, separator, count = raw.decode("utf-8", errors="replace").rpartition(":")
    if not separator or not incarnation or not count.isdigit():
        raise GenerationUnavailableError(f"write generation for {table_name!r} is malformed: {raw!r}")
    return incarnation, int(count)


class EpochGenerationSource:
    """reads and advances collection write generations in the epoch KV bucket."""

    def __init__(self, nats_client: KvCapable) -> None:
        """capture the client; no I/O.

        :param nats_client: connected client able to open the epoch bucket
        :ptype nats_client: KvCapable
        :return: nothing
        :rtype: None
        """
        self._nats = nats_client

    async def _bucket(self) -> KvBucketLike:
        """the epoch bucket, opened with no bucket-wide expiry so generations never lapse on a timer.

        :return: the bucket
        :rtype: KvBucketLike
        """
        return await self._nats.kv_bucket(name=_EPOCH_BUCKET, ttl=None)

    async def current(self, table_name: str) -> str:
        """the table's current write generation, minting its first incarnation when there is none.

        :param table_name: the collection's table
        :ptype table_name: str
        :return: the generation token, ``incarnation:count``
        :rtype: str
        :raises GenerationUnavailableError: when the generation cannot be read or established
        """
        key = generation_kv_key(table_name)
        try:
            bucket = await self._bucket()
            raw = await bucket.get(key=key)
            if raw is None:
                seeded = f"{uuid7()}:0".encode()  # convert at border: opaque KV value
                if await bucket.create(key=key, value=seeded) is not None:
                    raw = seeded
                else:
                    raw = await bucket.get(key=key)
        except KvError as exc:
            raise GenerationUnavailableError(f"write generation for {table_name!r} could not be read: {exc}") from exc
        if raw is None:
            # lost the create and read nothing: the bucket was emptied between the two.
            raise GenerationUnavailableError(f"write generation for {table_name!r} vanished while being established")
        _parse(table_name, raw)
        return raw.decode("utf-8")

    async def advance(self, table_name: str) -> str:
        """move the table's write generation on by one, and say what it became.

        The token returned is the value this call's own compare-and-swap wrote, never one a
        concurrent advance wrote after it: a writer stamps it on the row broadcasts of the commit
        it advanced for, and a follower counts those rows against exactly this advance.

        :param table_name: the collection's table
        :ptype table_name: str
        :return: the generation token this advance wrote, ``incarnation:count``
        :rtype: str
        :raises GenerationUnavailableError: when the generation cannot be advanced within the retry
            budget, or the store is unreachable
        """
        key = generation_kv_key(table_name)
        try:
            bucket = await self._bucket()
            for _ in range(_MAX_CAS_ATTEMPTS):
                entry = await bucket.get_entry(key=key)
                if entry is None:
                    # no generation yet (never read, or emptied): a new incarnation already
                    # invalidates every absence recorded under an old one, so starting at 1 is enough.
                    written = f"{uuid7()}:1"
                    if await bucket.create(key=key, value=written.encode()) is not None:
                        return written
                else:
                    raw, revision = entry
                    incarnation, count = _parse(table_name, raw)
                    written = f"{incarnation}:{count + 1}"
                    if await bucket.update(key=key, value=written.encode(), revision=revision) is not None:
                        return written
                await asyncio.sleep(random.uniform(0, _CAS_RETRY_BACKOFF_SECONDS))  # noqa: S311 - jitter, not security
        except KvError as exc:
            raise GenerationUnavailableError(
                f"write generation for {table_name!r} could not be advanced: {exc}"
            ) from exc
        raise GenerationUnavailableError(
            f"write generation for {table_name!r} lost {_MAX_CAS_ATTEMPTS} compare-and-swap rounds to concurrent writes"
        )


class EpochGenerationReader:
    """reads and watches collection write generations in the epoch bucket, and never writes them.

    What a pod that follows a table holds. It binds the bucket the hub declared rather than
    creating it, and where :meth:`EpochGenerationSource.current` mints a generation for a table
    that has none, this reports that there is none: a follower has no write to mint one with, and
    "no generation yet" is itself something its mark records.
    """

    def __init__(self, nats_client: KvCapable, *, create_if_missing: bool = False) -> None:
        """capture the client; no I/O.

        :param nats_client: connected client able to open the epoch bucket
        :ptype nats_client: KvCapable
        :param create_if_missing: whether opening the bucket may create it. ``False``, the default,
            binds only, which is all a pod's grant allows; ``True`` is for a principal that also
            declares the bucket and follows through the same reader
        :ptype create_if_missing: bool
        :return: nothing
        :rtype: None
        """
        self._nats = nats_client
        self._create_if_missing = create_if_missing

    async def _bucket(self) -> KvBucketLike:
        """the epoch bucket, opened as :class:`EpochGenerationSource` opens it but bound, not created.

        :return: the bucket
        :rtype: KvBucketLike
        """
        return await self._nats.kv_bucket(name=_EPOCH_BUCKET, ttl=None, create_if_missing=self._create_if_missing)

    async def read(self, table_name: str) -> str | None:
        """the table's current write generation, or ``None`` when the bucket holds none for it.

        :param table_name: the collection's table
        :ptype table_name: str
        :return: the generation token ``incarnation:count``, or ``None``
        :rtype: str | None
        :raises GenerationUnavailableError: when the bucket cannot be read, or holds a value that
            is not a generation

        A read is a get of the key, which the whole-bucket read an agent pod holds admits. A
        principal granted single keys only (``JsCapability.KV_KEY_READ``) reaches them through
        :meth:`watch` instead: that grant's get is the direct one, which the epoch bucket is not
        declared for.
        """
        key = generation_kv_key(table_name)
        try:
            bucket = await self._bucket()
            raw = await bucket.get(key=key)
        except KvError as exc:
            raise GenerationUnavailableError(f"write generation for {table_name!r} could not be read: {exc}") from exc
        if raw is None:
            return None
        _parse(table_name, raw)
        return raw.decode("utf-8")

    async def watch(self, table_name: str) -> AsyncGenerator[str | None]:
        """the table's write generation as it stands, then each value it takes, until the caller stops.

        Pushed by the broker through one named consumer on the table's key
        (:meth:`threetears.nats.NatsKvBucket.watch_key`), which a grant narrowed to that one key
        admits. The first value yielded is the key's latest, so a watcher that starts, or restarts
        after the broker did, needs no separate read; a key that has never been written yields
        nothing until it is. A value that is not a generation is yielded as it is: it compares
        unequal to every mark, which is what makes a follower drop the table for it.

        Close it by stopping iteration.

        :param table_name: the collection's table
        :ptype table_name: str
        :return: the generation tokens, in order; ``None`` when the key was deleted
        :rtype: AsyncGenerator[str | None]
        :raises GenerationUnavailableError: when the bucket cannot be opened or cannot watch a key
        :raises KvError: when the connection closes, so nothing can ever be delivered again
        """
        key = generation_kv_key(table_name)
        try:
            bucket = await self._bucket()
        except KvError as exc:
            raise GenerationUnavailableError(
                f"write generation for {table_name!r} could not be watched: {exc}"
            ) from exc
        if not isinstance(bucket, _KeyWatchable):
            raise GenerationUnavailableError(
                f"write generation for {table_name!r} could not be watched: {type(bucket).__name__} has no watch_key"
            )
        async with aclosing(bucket.watch_key(key=key)) as updates:
            async for update in updates:
                value: bytes | None = update.value
                yield None if value is None else value.decode("utf-8", errors="replace")
