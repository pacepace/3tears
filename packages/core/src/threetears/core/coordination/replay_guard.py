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

FAIL-CLOSED, unlike the L2 cache accessors on :class:`~threetears.core.collections.BaseCollection`
(deliberately fail-open): a transport failure that the backing
:class:`~threetears.nats.kv.KvBucketLike` cannot self-heal propagates as
:class:`~threetears.nats.KvError`, so the caller DENIES rather than silently admitting a possible
replay.

**A wiped bucket fails closed too.** The bucket is memory-backed, so a broker restart empties it
and the wrapper's self-heal recreates it -- and any handle, on any replica, then keeps working
silently against the empty bucket. A nonce recorded before the wipe is no longer there to refuse
its replay. So a fresh record also reads the bucket's creation time from the server and refuses
any artifact issued before it: a replay can only find its nonce missing if the bucket was wiped
after the original was accepted, which puts the original's signed issue time before the new
creation time. The read comes AFTER the create, so a wipe racing the check can only make it
stricter.

**How far past the creation time the refusal reaches.** A verifier accepts an issue time up to its
own future tolerance ahead of ITS clock, and the creation time comes from the BROKER's clock. So
the refusal must reach the verifier's future tolerance plus the drift between those two hosts, or
a replay stamped at the edge of acceptance could slip past. The guard is given the verifier's
future tolerance and adds :data:`CLOCK_DRIFT_ALLOWANCE` itself, so the drift is modelled in one
place rather than guessed at each call site. A verifier calls :meth:`ReplayGuard.require_covers`
with its own tolerance, so a leeway widened later fails loudly instead of reopening the hole.

The cost is that for that long after a wipe, fresh artifacts are refused too -- the price of never
admitting a replay.

    guard = ReplayGuard(
        nats_client, bucket_name="pop_nonces", ttl_seconds=120,
        verifier_future_tolerance=timedelta(seconds=60),
    )
    guard.require_covers(timedelta(seconds=leeway_seconds))  # at the verifier's construction
    if not await guard.record_unique(nonce, issued_at=proof_issued_at):
        raise <replay rejected>
"""

from __future__ import annotations

import asyncio
import hashlib
from datetime import datetime, timedelta
from typing import TYPE_CHECKING

from threetears.observe import get_logger

if TYPE_CHECKING:
    # From the submodule, not the package: these three are Protocols that
    # `threetears.nats` stopped re-exporting when its nats-py-backed surface went lazy.
    # Annotation-only, so the eager `kv` import here costs an L1 consumer nothing.
    from threetears.nats.kv import KvBucketLike, KvCapable

__all__ = ["CLOCK_DRIFT_ALLOWANCE", "ReplayGuard", "RevocationGuard"]

log = get_logger(__name__)

#: how far the clocks of two NTP-synchronised platform hosts -- a verifier and the NATS broker -- may
#: disagree. Added to every guard's verifier future tolerance, because the creation time the wipe
#: check compares against is the broker's clock while the tolerance is measured on the verifier's.
#: Every second of it is also a second of refused traffic after a broker restart.
CLOCK_DRIFT_ALLOWANCE = timedelta(seconds=5)


class ReplayGuard:
    """records single-use nonces in a shared, TTL'd KV bucket; rejects any second sighting."""

    def __init__(
        self,
        nats_client: "KvCapable",
        *,
        bucket_name: str,
        ttl_seconds: int,
        verifier_future_tolerance: timedelta,
    ) -> None:
        """configure the guard; defer bucket binding until the first record.

        :param nats_client: connected canonical :class:`threetears.nats.kv.KvCapable`; the guard
            opens its KV bucket through :meth:`KvCapable.kv_bucket`
        :ptype nats_client: KvCapable
        :param bucket_name: KV bucket suffix; the wrapper prefixes it with the namespace. Pick a
            bucket dedicated to one assertion kind (e.g. ``pop_nonces``) so unrelated nonces never
            collide across surfaces
        :ptype bucket_name: str
        :param ttl_seconds: how long a recorded nonce is remembered — set equal to the assertion's
            accept window. MUST be positive: a non-positive TTL would mean entries never expire,
            growing the bucket without bound
        :ptype ttl_seconds: int
        :param verifier_future_tolerance: how far ahead of the verifier's clock the verifier
            accepts an artifact's issue time. The guard refuses, after a wipe, anything issued within
            this plus :data:`CLOCK_DRIFT_ALLOWANCE` of the bucket's creation time. Deliberately has
            no default: it is a property of the verifier, which confirms it with
            :meth:`require_covers`. MUST NOT be negative
        :ptype verifier_future_tolerance: timedelta
        :raises ValueError: when ``ttl_seconds`` is not positive or the tolerance is negative
        """
        if ttl_seconds <= 0:
            raise ValueError(f"ReplayGuard ttl_seconds must be positive, got {ttl_seconds}")
        if verifier_future_tolerance < timedelta(0):
            raise ValueError(
                f"ReplayGuard verifier_future_tolerance must not be negative, got {verifier_future_tolerance}"
            )
        self._client = nats_client
        self._bucket_name = bucket_name
        self._ttl = timedelta(seconds=ttl_seconds)
        self._verifier_future_tolerance = verifier_future_tolerance
        self._bucket: "KvBucketLike | None" = None
        self._bucket_lock = asyncio.Lock()

    @property
    def bucket_name(self) -> str:
        """the configured bucket suffix.

        :return: bucket name
        :rtype: str
        """
        return self._bucket_name

    @property
    def verifier_future_tolerance(self) -> timedelta:
        """the verifier future tolerance this guard's wipe check was sized for.

        :return: the configured tolerance, excluding the drift allowance
        :rtype: timedelta
        """
        return self._verifier_future_tolerance

    def require_covers(self, future_tolerance: timedelta) -> None:
        """refuse to serve a verifier whose future tolerance this guard was not sized for.

        A verifier that accepts issue times further ahead than the guard expects would let a
        replay stamped at that edge past the wipe check, with nothing to say so. Calling this where
        the verifier is configured turns that silent hole into a startup failure.

        :param future_tolerance: how far ahead of its clock the calling verifier accepts an issue
            time
        :ptype future_tolerance: timedelta
        :return: None
        :rtype: None
        :raises ValueError: when ``future_tolerance`` exceeds this guard's configured tolerance
        """
        if future_tolerance > self._verifier_future_tolerance:
            raise ValueError(
                f"ReplayGuard {self._bucket_name!r} was sized for a verifier future tolerance of "
                f"{self._verifier_future_tolerance}, but its verifier accepts issue times up to "
                f"{future_tolerance} ahead; a replay stamped at that edge would pass the wipe check. "
                "Construct the guard with verifier_future_tolerance at least the verifier's leeway."
            )

    async def record_unique(self, nonce: str, *, issued_at: datetime) -> bool:
        """record ``nonce``; return ``True`` if FRESH, ``False`` if it must be REFUSED.

        Refused means either already recorded (a replay), or issued before the bucket's current
        incarnation began, allowing for the verifier's future tolerance and
        :data:`CLOCK_DRIFT_ALLOWANCE` -- an artifact whose earlier sighting a wipe may have erased.
        The nonce is recorded either way, so a refused artifact stays refused.

        The nonce is hashed into a fixed, KV-safe key, so any nonce format is accepted and the raw
        nonce is never stored as a key. Backed by CAS create-if-absent, so the fresh/replay
        decision is atomic even under concurrent calls on different replicas.

        :param nonce: the per-assertion nonce to consume
        :ptype nonce: str
        :param issued_at: no later than the earliest moment the artifact could first have been
            accepted -- normally its signed issue time. MUST be timezone-aware
        :ptype issued_at: datetime
        :return: ``True`` when the artifact may be used, ``False`` when it must be refused
        :rtype: bool
        :raises ValueError: when ``issued_at`` is timezone-naive
        :raises threetears.nats.KvError: on a KV transport failure, or when the bucket's creation
            time cannot be read — the caller MUST treat this as a failed check and DENY
            (fail-closed), never as fresh
        """
        if issued_at.tzinfo is None:
            raise ValueError("ReplayGuard.record_unique requires a timezone-aware issued_at")
        bucket = await self._ensure_bucket()
        revision = await bucket.create(key=self._key(nonce), value=b"1")
        fresh = revision is not None  # None == key already existed == replay
        if fresh:
            # read only after the create: a wipe landing between the two can only move this
            # later, which refuses more, never less.
            date_created = await bucket.date_created()
            refusal_reach = self._verifier_future_tolerance + CLOCK_DRIFT_ALLOWANCE
            if issued_at < date_created + refusal_reach:
                log.warning(
                    "ReplayGuard refused an artifact issued before its bucket was created; "
                    "the bucket was wiped or is new, so an earlier sighting cannot be ruled out",
                    extra={
                        "extra_data": {
                            "bucket": self._bucket_name,
                            "issued_at": issued_at.isoformat(),
                            "bucket_date_created": date_created.isoformat(),
                            "refusal_reach_seconds": refusal_reach.total_seconds(),
                        }
                    },
                )
                fresh = False
        return fresh

    async def _ensure_bucket(self) -> "KvBucketLike":
        """open (or bind) the TTL'd KV bucket once; async-safe lazy init."""
        if self._bucket is not None:
            return self._bucket
        async with self._bucket_lock:
            if self._bucket is None:
                # memory storage: a wipe is detected by record_unique's creation-time check, not
                # survived. see the module docstring.
                self._bucket = await self._client.kv_bucket(
                    name=self._bucket_name,
                    ttl=self._ttl,
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

    def __init__(self, nats_client: "KvCapable", *, bucket_name: str, ttl_seconds: int) -> None:
        """configure the guard; defer bucket binding until the first record.

        :param nats_client: connected canonical :class:`threetears.nats.kv.KvCapable`; the guard
            opens its KV bucket through :meth:`KvCapable.kv_bucket`
        :ptype nats_client: KvCapable
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
        self._bucket: "KvBucketLike | None" = None
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

    async def _ensure_bucket(self) -> "KvBucketLike":
        """open (or bind) the TTL'd KV bucket once; async-safe lazy init."""
        if self._bucket is not None:
            return self._bucket
        async with self._bucket_lock:
            if self._bucket is None:
                self._bucket = await self._client.kv_bucket(
                    name=self._bucket_name,
                    ttl=self._ttl,
                    # file storage: a revocation entry is a security control, not a cache -- a
                    # restart must not silently forget an active revocation. temporary: this
                    # state belongs in L3, see docs/design-durable-coordination.md.
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
