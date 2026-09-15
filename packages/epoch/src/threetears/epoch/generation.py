"""collection write generations over the epoch bucket.

Implements :class:`threetears.core.collections.generation.GenerationSource`, which a collection that
caches absences stamps every recorded absence with.

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
from typing import TYPE_CHECKING, Final

from threetears.core.exceptions import GenerationUnavailableError
from threetears.nats.errors import KvError
from threetears.nats.subjects import Subjects
from threetears.observe import get_logger
from uuid_utils import uuid7

from threetears.epoch.client import _EPOCH_BUCKET, _key_for

if TYPE_CHECKING:
    from threetears.nats.kv import KvBucketLike, KvCapable

__all__ = ["EpochGenerationSource"]

log = get_logger(__name__)

#: matched to the coordination counters' budget: contention on one table's generation is a burst of
#: concurrent writes to that table, which is exactly when a small budget would give up.
_MAX_CAS_ATTEMPTS: Final = 30

#: full-jitter backoff bound between compare-and-swap retries, seconds.
_CAS_RETRY_BACKOFF_SECONDS: Final = 0.02


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
        key = _key_for(Subjects.collection_generation_epoch(table_name))
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

    async def advance(self, table_name: str) -> None:
        """move the table's write generation on by one.

        :param table_name: the collection's table
        :ptype table_name: str
        :return: None
        :rtype: None
        :raises GenerationUnavailableError: when the generation cannot be advanced within the retry
            budget, or the store is unreachable
        """
        key = _key_for(Subjects.collection_generation_epoch(table_name))
        try:
            bucket = await self._bucket()
            for _ in range(_MAX_CAS_ATTEMPTS):
                entry = await bucket.get_entry(key=key)
                if entry is None:
                    # no generation yet (never read, or emptied): a new incarnation already
                    # invalidates every absence recorded under an old one, so starting at 1 is enough.
                    if await bucket.create(key=key, value=f"{uuid7()}:1".encode()) is not None:
                        return
                else:
                    raw, revision = entry
                    incarnation, count = _parse(table_name, raw)
                    if (
                        await bucket.update(key=key, value=f"{incarnation}:{count + 1}".encode(), revision=revision)
                        is not None
                    ):
                        return
                await asyncio.sleep(random.uniform(0, _CAS_RETRY_BACKOFF_SECONDS))  # noqa: S311 - jitter, not security
        except KvError as exc:
            raise GenerationUnavailableError(
                f"write generation for {table_name!r} could not be advanced: {exc}"
            ) from exc
        raise GenerationUnavailableError(
            f"write generation for {table_name!r} lost {_MAX_CAS_ATTEMPTS} compare-and-swap rounds to concurrent writes"
        )
