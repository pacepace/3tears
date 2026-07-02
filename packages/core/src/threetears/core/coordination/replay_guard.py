"""single-use nonce guard backed by NATS JetStream KV — replay protection for signed assertions.

A proof-of-possession (agent→proxy) or proxy→pod assertion carries a nonce that must be honoured
at most once. :class:`ReplayGuard` records each nonce with CAS create-if-absent in a TTL'd KV
bucket SHARED across every replica: the first caller to record a nonce wins (fresh); any later
caller — a replay, possibly landing on a different replica — finds the key already present and is
rejected. An in-process set could not do this: registry/tool-pod replicas are load-balanced, so a
replay simply lands on a replica that never saw the original.

The bucket TTL bounds both memory and the replay window: a nonce only needs remembering for as
long as the assertion carrying it is still inside its accept window. Set ``ttl_seconds`` to that
window.

FAIL-CLOSED, unlike the L2 cache facade (:class:`threetears.core.cache.kv.NatsKvClient`, which is
deliberately fail-open): a transport failure that the backing
:class:`~threetears.nats.NatsKvBucket` cannot self-heal propagates as
:class:`~threetears.nats.KvError`, so the caller DENIES rather than silently admitting a possible
replay. The bucket uses ``file`` storage so a normal NATS restart re-binds the intact on-disk
bucket and recorded nonces survive within their TTL (with the default ``memory`` storage a restart
wipes the stream and the bucket's self-heal would recreate it EMPTY -- a fail-open replay hole).
The one residual: total stream/disk loss resets the replay window for at most one accept-window;
that is unavoidable and bounded.

    guard = ReplayGuard(nats_client, bucket_name="pop_nonces", ttl_seconds=120)
    if not await guard.record_unique(nonce):
        raise <replay rejected>
"""

from __future__ import annotations

import asyncio
import hashlib
from datetime import timedelta
from typing import TYPE_CHECKING

from threetears.observe import get_logger

if TYPE_CHECKING:
    from threetears.nats import NatsClient, NatsKvBucket

__all__ = ["ReplayGuard"]

log = get_logger(__name__)


class ReplayGuard:
    """records single-use nonces in a shared, TTL'd KV bucket; rejects any second sighting."""

    def __init__(self, nats_client: "NatsClient", *, bucket_name: str, ttl_seconds: int) -> None:
        """configure the guard; defer bucket binding until the first record.

        :param nats_client: connected canonical :class:`threetears.nats.NatsClient`; the guard
            opens its KV bucket through :meth:`NatsClient.kv_bucket`
        :ptype nats_client: NatsClient
        :param bucket_name: KV bucket suffix; the wrapper prefixes it with the namespace. Pick a
            bucket dedicated to one assertion kind (e.g. ``pop_nonces``) so unrelated nonces never
            collide across surfaces
        :ptype bucket_name: str
        :param ttl_seconds: how long a recorded nonce is remembered — set equal to the assertion's
            accept window. MUST be positive: a non-positive TTL would mean entries never expire,
            growing the bucket without bound
        :ptype ttl_seconds: int
        :raises ValueError: when ``ttl_seconds`` is not positive
        """
        if ttl_seconds <= 0:
            raise ValueError(f"ReplayGuard ttl_seconds must be positive, got {ttl_seconds}")
        self._client = nats_client
        self._bucket_name = bucket_name
        self._ttl = timedelta(seconds=ttl_seconds)
        self._bucket: "NatsKvBucket | None" = None
        self._bucket_lock = asyncio.Lock()

    @property
    def bucket_name(self) -> str:
        """the configured bucket suffix.

        :return: bucket name
        :rtype: str
        """
        return self._bucket_name

    async def record_unique(self, nonce: str) -> bool:
        """record ``nonce``; return ``True`` if FRESH, ``False`` if a REPLAY (already recorded).

        The nonce is hashed into a fixed, KV-safe key, so any nonce format is accepted and the raw
        nonce is never stored as a key. Backed by CAS create-if-absent, so the fresh/replay
        decision is atomic even under concurrent calls on different replicas.

        :param nonce: the per-assertion nonce to consume
        :ptype nonce: str
        :return: ``True`` on first sighting (recorded), ``False`` if it was already recorded
        :rtype: bool
        :raises threetears.nats.KvError: on a KV transport failure — the caller MUST treat this as
            a failed check and DENY (fail-closed), never as fresh
        """
        bucket = await self._ensure_bucket()
        revision = await bucket.create(key=self._key(nonce), value=b"1")
        return revision is not None  # None == key already existed == replay

    async def _ensure_bucket(self) -> "NatsKvBucket":
        """open (or bind) the TTL'd KV bucket once; async-safe lazy init."""
        if self._bucket is not None:
            return self._bucket
        async with self._bucket_lock:
            if self._bucket is None:
                self._bucket = await self._client.kv_bucket(
                    name=self._bucket_name,
                    ttl=self._ttl,
                    # file storage (NOT the default "memory"): the replay cache must survive a NATS
                    # restart. with memory storage a restart wipes the stream, and NatsKvBucket's
                    # self-heal (kv.py _run_with_reopen) would recreate the bucket EMPTY -- so a
                    # pre-restart nonce would be admitted as "fresh" for one accept-window, a
                    # fail-OPEN replay hole. file storage rebinds the intact on-disk bucket instead,
                    # so recorded nonces persist within their TTL across a normal restart.
                    storage="file",
                    create_if_missing=True,
                    history=1,
                )
                log.info("ReplayGuard bound bucket %s", self._bucket_name)
        return self._bucket

    @staticmethod
    def _key(nonce: str) -> str:
        """hash the nonce into a fixed-length, KV-safe key (also avoids storing the raw nonce)."""
        return hashlib.sha256(nonce.encode("utf-8")).hexdigest()
