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
from datetime import datetime, timedelta
from typing import TYPE_CHECKING

from threetears.observe import get_logger

if TYPE_CHECKING:
    from threetears.nats import NatsClient, NatsKvBucket

__all__ = ["ReplayGuard", "RevocationGuard"]

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


class RevocationGuard:
    """timestamped revocation entries in a shared, TTL'd KV bucket -- a sibling to
    :class:`ReplayGuard`, not a replacement.

    :meth:`ReplayGuard.record_unique` stores a presence-only sentinel: a bare "have I seen this
    key" membership test is the right shape for a single-use nonce, and is also the right shape
    for revoking one specific ``jti`` (a single token) or ``sid`` (a whole refresh-token family) --
    "is *this exact* key on the denylist" needs no timestamp, so those two key shapes are recorded
    with plain :class:`ReplayGuard` and this class is not involved.

    A revoked ``sub`` (principal) or ``customer_id`` (tenant) entry is different in kind, not just
    in name: the denylist check is NOT bare membership. A revocation recorded against a ``sub`` or
    ``customer_id`` must block every session that STARTED BEFORE the revocation, while a session
    that starts AFTER it is a legitimate new session and must be unaffected -- otherwise every
    future login to a once-revoked principal or tenant would be permanently denylisted, which is
    not what recovery/offboarding mean. So the entry has to carry a VALUE (the moment of
    revocation), and the check compares that value against a caller-supplied moment, rather than
    just testing presence. See ``data-model.md``'s "Revocation denylist entries":
    ``session_started_at < revoked_at``.

        guard = RevocationGuard(nats_client, bucket_name="revocations", ttl_seconds=...)
        await guard.record_revocation("sub:<principal_id>", revoked_at=cutoff)
        blocked = await guard.is_revoked_before("sub:<principal_id>", moment=session_started_at)
    """

    def __init__(self, nats_client: "NatsClient", *, bucket_name: str, ttl_seconds: int) -> None:
        """configure the guard; defer bucket binding until the first record.

        :param nats_client: connected canonical :class:`threetears.nats.NatsClient`; the guard
            opens its KV bucket through :meth:`NatsClient.kv_bucket`
        :ptype nats_client: NatsClient
        :param bucket_name: KV bucket suffix; the wrapper prefixes it with the namespace. Pick a
            bucket dedicated to revocation entries, distinct from any :class:`ReplayGuard` bucket
            sharing the same process, so the two key shapes never collide
        :ptype bucket_name: str
        :param ttl_seconds: how long a recorded revocation is remembered. MUST be at least as long
            as the longest session lifetime a revocation needs to outlive (a revocation entry that
            expires before every session it denylists has naturally ended would fail OPEN). MUST be
            positive: a non-positive TTL would mean entries never expire, growing the bucket without
            bound
        :ptype ttl_seconds: int
        :raises ValueError: when ``ttl_seconds`` is not positive
        """
        if ttl_seconds <= 0:
            raise ValueError(f"RevocationGuard ttl_seconds must be positive, got {ttl_seconds}")
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

    async def record_revocation(self, key: str, *, revoked_at: datetime) -> None:
        """record (or overwrite) the revoked-at timestamp for ``key``.

        Unconditional write, not create-if-absent: a second revocation call for the same key (an
        operator re-running tenant offboarding, or narrowing an earlier cutoff) replaces the
        effective revocation moment with the new value rather than being rejected as a duplicate.

        :param key: the denylist key -- e.g. ``f"sub:{principal_id}"`` or
            ``f"customer_id:{customer_id}"``. Hashed into a fixed, KV-safe key before storage, so
            any key format is accepted and the raw identifier is never stored as a KV key
        :ptype key: str
        :param revoked_at: the moment the key is considered revoked from. MUST be timezone-aware --
            a naive datetime cannot be compared reliably against a caller-supplied moment that may
            come from a different clock/offset assumption
        :ptype revoked_at: datetime
        :raises ValueError: when ``revoked_at`` is timezone-naive
        :raises threetears.nats.KvError: on a KV transport failure
        """
        if revoked_at.tzinfo is None:
            raise ValueError("RevocationGuard.record_revocation requires a timezone-aware revoked_at")
        bucket = await self._ensure_bucket()
        await bucket.put(key=self._key(key), value=revoked_at.isoformat().encode("utf-8"))

    async def revoked_at(self, key: str) -> datetime | None:
        """return the stored revocation timestamp for ``key``, or ``None`` if never revoked.

        :param key: the denylist key to look up
        :ptype key: str
        :return: the recorded revocation moment, or ``None`` if ``key`` has no revocation entry
        :rtype: datetime | None
        :raises threetears.nats.KvError: on a KV transport failure
        """
        bucket = await self._ensure_bucket()
        raw = await bucket.get(key=self._key(key))
        if raw is None:
            return None
        return datetime.fromisoformat(raw.decode("utf-8"))

    async def is_revoked_before(self, key: str, *, moment: datetime) -> bool:
        """return whether ``key``'s recorded revocation denylists something that started at
        ``moment``.

        ``True`` iff a revocation is recorded for ``key`` AND ``moment < revoked_at`` -- the
        ``session_started_at < revoked_at`` check from ``data-model.md``'s "Revocation denylist
        entries": something that started BEFORE the revocation is always denylisted; something
        that starts AT OR AFTER it is unaffected. Returns ``False`` (never raises) when ``key`` has
        no revocation entry -- that is the legitimate "not revoked" case, not a failure; a KV
        transport failure still propagates as :class:`~threetears.nats.KvError` and the caller MUST
        treat that as deny (fail-closed on infrastructure failure, never on absence).

        :param key: the denylist key to check
        :ptype key: str
        :param moment: the comparison timestamp (e.g. a session's ``session_started_at``). MUST be
            timezone-aware, for the same reason as :meth:`record_revocation`'s ``revoked_at``
        :ptype moment: datetime
        :return: ``True`` if ``key`` is revoked as of ``moment``, ``False`` otherwise
        :rtype: bool
        :raises ValueError: when ``moment`` is timezone-naive
        :raises threetears.nats.KvError: on a KV transport failure -- the caller MUST deny
        """
        if moment.tzinfo is None:
            raise ValueError("RevocationGuard.is_revoked_before requires a timezone-aware moment")
        stored = await self.revoked_at(key)
        if stored is None:
            return False
        return moment < stored

    async def _ensure_bucket(self) -> "NatsKvBucket":
        """open (or bind) the TTL'd KV bucket once; async-safe lazy init."""
        if self._bucket is not None:
            return self._bucket
        async with self._bucket_lock:
            if self._bucket is None:
                self._bucket = await self._client.kv_bucket(
                    name=self._bucket_name,
                    ttl=self._ttl,
                    # file storage: a revocation entry is a security control, not a cache -- a
                    # restart must not silently forget an active revocation (the same fail-open
                    # concern ReplayGuard's bucket comment documents).
                    storage="file",
                    create_if_missing=True,
                    history=1,
                )
                log.info("RevocationGuard bound bucket %s", self._bucket_name)
        return self._bucket

    @staticmethod
    def _key(key: str) -> str:
        """hash the denylist key into a fixed-length, KV-safe key (also avoids storing it raw)."""
        return hashlib.sha256(key.encode("utf-8")).hexdigest()
