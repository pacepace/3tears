"""The checkpoint saver's L2 cache over NATS KV.

:class:`NatsCheckpointL2Cache` connects
:class:`~threetears.langgraph.checkpoint.ThreeTierCheckpointSaver` to a
:class:`~threetears.nats.NatsClient`'s KV buckets and implements every L2
protocol the saver checks for:

- :class:`~threetears.langgraph.protocols.CheckpointL2Cache` -- get, put, delete;
- :class:`~threetears.langgraph.protocols.CheckpointL2RevisionCache` -- the
  revision read and revision-fenced write that keep a refill after an L3 read from
  replacing a newer checkpoint;
- :class:`~threetears.langgraph.protocols.CheckpointL2ValueLimit` -- the server's
  advertised ``max_payload``, so a checkpoint too big for the bucket is never put;
- :class:`~threetears.langgraph.protocols.CheckpointL2PrefixCache` -- a prefix
  sweep, so a purge clears every namespaced bundle of a thread.

Needs ``3tears-nats``: install ``3tears-langgraph[nats]``. The module is not
imported by ``threetears.langgraph`` itself, so a saver without NATS never needs it.

The saver owns key shaping and the bucket name; this adapter only opens the
named bucket (``NatsClient.kv_bucket``, which prefixes the namespace and caches
the handle) and runs the operation. Failures raise, and the saver contains them.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from threetears.langgraph.protocols import (
    CheckpointL2Cache,
    CheckpointL2PrefixCache,
    CheckpointL2RevisionCache,
    CheckpointL2ValueLimit,
)

if TYPE_CHECKING:
    from threetears.nats import NatsClient

__all__ = ["NatsCheckpointL2Cache"]


class NatsCheckpointL2Cache(
    CheckpointL2Cache, CheckpointL2RevisionCache, CheckpointL2ValueLimit, CheckpointL2PrefixCache
):
    """the checkpoint L2 cache over a :class:`~threetears.nats.NatsClient`'s KV buckets.

    :param client: the connected NATS client whose buckets hold the cache
    :ptype client: NatsClient
    """

    def __init__(self, client: NatsClient) -> None:
        """capture the client.

        :param client: the connected NATS client
        :ptype client: NatsClient
        :return: nothing
        :rtype: None
        """
        self._client = client

    @property
    def max_value_bytes(self) -> int | None:
        """the connected server's ``max_payload``, the largest value a put can carry.

        :return: the limit in bytes, or None while no server has advertised one
        :rtype: int | None
        """
        return self._client.max_payload

    async def get(self, bucket: str, key: str) -> bytes | None:
        """read a value.

        :param bucket: bucket name suffix (the client prefixes its namespace)
        :ptype bucket: str
        :param key: cache key, shaped by the saver
        :ptype key: str
        :return: the value, or None when the key holds none
        :rtype: bytes | None
        """
        return await (await self._client.kv_bucket(name=bucket)).get(key=key)

    async def put(self, bucket: str, key: str, value: bytes) -> None:
        """write a value unconditionally.

        :param bucket: bucket name suffix
        :ptype bucket: str
        :param key: cache key
        :ptype key: str
        :param value: serialized checkpoint bundle
        :ptype value: bytes
        :return: nothing
        :rtype: None
        """
        await (await self._client.kv_bucket(name=bucket)).put(key=key, value=value)

    async def delete(self, bucket: str, key: str) -> None:
        """delete a key; a key that holds nothing is not an error.

        :param bucket: bucket name suffix
        :ptype bucket: str
        :param key: cache key
        :ptype key: str
        :return: nothing
        :rtype: None
        """
        await (await self._client.kv_bucket(name=bucket)).delete(key=key)

    async def get_latest(self, bucket: str, key: str) -> tuple[bytes | None, int]:
        """the value and the key's latest revision, a deletion counted as a revision.

        :param bucket: bucket name suffix
        :ptype bucket: str
        :param key: cache key
        :ptype key: str
        :return: the value or None, and the revision (``0`` for a key never written)
        :rtype: tuple[bytes | None, int]
        """
        return await (await self._client.kv_bucket(name=bucket)).get_latest(key=key)

    async def update(self, bucket: str, key: str, value: bytes, revision: int) -> int | None:
        """write only while the key's latest revision is ``revision``.

        :param bucket: bucket name suffix
        :ptype bucket: str
        :param key: cache key
        :ptype key: str
        :param value: serialized checkpoint bundle
        :ptype value: bytes
        :param revision: the revision the key must still be at
        :ptype revision: int
        :return: the new revision, or None when the key changed since
        :rtype: int | None
        """
        return await (await self._client.kv_bucket(name=bucket)).update(key=key, value=value, revision=revision)

    async def delete_prefix(self, bucket: str, prefix: str) -> None:
        """delete every live key that starts with ``prefix``.

        :param bucket: bucket name suffix
        :ptype bucket: str
        :param prefix: key prefix, separator included (the saver sweeps ``<thread>.``)
        :ptype prefix: str
        :return: nothing
        :rtype: None
        """
        kv = await self._client.kv_bucket(name=bucket)
        for key in await kv.list_keys(prefix=prefix):
            await kv.delete(key=key)
