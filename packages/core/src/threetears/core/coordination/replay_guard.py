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

**Bind at start, or the window is measured from the wrong moment.** The watermark is measured from
the bucket's creation time, and the bucket is created by whichever call opens it first. A service
that calls :meth:`ReplayGuard.bind` at startup creates it before it serves anything, so after a
wipe it refuses only what was issued before it started, or within the reach of its start. One
that leaves the open to its first :meth:`ReplayGuard.record_unique` creates the bucket at first
USE, and refuses the artifact that arrived first however long after startup it was issued.

**A first run is not a wipe, and an anchor is what tells them apart.** Without one the guard
cannot distinguish a bucket it has never had from one it lost, so it assumes the worse and pays
that cost on a fresh deployment too -- where nothing was ever recorded and no replay is possible.
Pass a :class:`~threetears.core.coordination.replay_anchor.ReplayAnchor` and the watermark
applies only when the anchor predates the bucket. Consumers holding durable storage should; one
that holds only a NATS client keeps today's conservative behaviour by leaving it unset.

    guard = ReplayGuard(
        nats_client, bucket_name="pop_nonces", ttl_seconds=120,
        verifier_future_tolerance=timedelta(seconds=60),
    )
    guard.require_covers(timedelta(seconds=leeway_seconds))  # at the verifier's construction
    await guard.bind()  # at service start, before serving anything
    if not await guard.record_unique(nonce, issued_at=proof_issued_at):
        raise <replay rejected>
"""

from __future__ import annotations

import asyncio
import hashlib
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

from threetears.observe import get_logger

if TYPE_CHECKING:
    # From the submodule, not the package: these three are Protocols that
    # `threetears.nats` stopped re-exporting when its nats-py-backed surface went lazy.
    # Annotation-only, so the eager `kv` import here costs an L1 consumer nothing.
    from threetears.nats.kv import KvBucketLike, KvCapable

    from threetears.core.coordination.replay_anchor import ReplayAnchor

__all__ = ["CLOCK_DRIFT_ALLOWANCE", "ReplayGuard"]

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
        anchor: "ReplayAnchor | None" = None,
    ) -> None:
        """configure the guard; the bucket is opened by :meth:`bind`, which a service calls at start.

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
        :param anchor: durable record of when this ledger FIRST existed
            (:mod:`threetears.core.coordination.replay_anchor`). Without one the guard cannot
            tell a first run from a wipe and applies the watermark to both -- correct after a
            wipe, and on a first run a window of refusals protecting nothing. With one, the
            watermark applies only when the anchor predates the bucket, which is what a wipe
            looks like. Optional because the registry server and the tool pod deliberately hold
            only a NATS client, and a minute of refused internal RPC that retries does not
            justify wiring durable storage into them
        :ptype anchor: ReplayAnchor | None
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
        self._anchor = anchor
        # Read once, at the first record, and kept: the anchor is a fact about this ledger's
        # whole history, so re-reading it per artifact would put a durable round trip on the
        # hot path to learn something that cannot change while the process runs.
        self._ledger_first_existed: datetime | None = None
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
            time cannot be read -- the caller MUST treat this as a failed check and DENY
            (fail-closed), never as fresh
        """
        if issued_at.tzinfo is None:
            raise ValueError("ReplayGuard.record_unique requires a timezone-aware issued_at")
        bucket = await self.bind()
        revision = await bucket.create(key=self._key(nonce), value=b"1")
        fresh = revision is not None  # None == key already existed == replay
        if fresh:
            # read only after the create: a wipe landing between the two can only move this
            # later, which refuses more, never less.
            date_created = await bucket.date_created()
            refusal_reach = self._verifier_future_tolerance + CLOCK_DRIFT_ALLOWANCE
            if await self._bucket_replaced_a_lost_one(date_created) and issued_at < date_created + refusal_reach:
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

    async def _bucket_replaced_a_lost_one(self, date_created: datetime) -> bool:
        """whether this bucket replaced an earlier one whose nonces were lost.

        The watermark exists for that case alone. Without an anchor the guard cannot tell it
        from a first run and must assume it, which is what makes a fresh deployment refuse
        artifacts it has no reason to doubt.

        **Every failure answers ``True``.** An anchor that cannot be read, or written, leaves the
        guard exactly as blind as having none, and the blind answer is the conservative one. It
        must never be the permissive one: an anchor whose write silently failed would read as
        absent forever, and a guard that treated absence as proof of a first run would skip the
        watermark after every wipe.

        :param date_created: the bucket's creation time, from the broker's clock
        :ptype date_created: datetime
        :return: whether the watermark should be applied
        :rtype: bool
        """
        replaced = True
        if self._anchor is not None:
            if self._ledger_first_existed is None:
                try:
                    self._ledger_first_existed = await self._anchor.first_existed(
                        self._bucket_name, now=datetime.now(UTC)
                    )
                except Exception as exc:  # prawduct:allow prawduct/broad-except -- see below
                    # Deliberately broad, and narrow in effect: an anchor is a Protocol, so the
                    # storage failures an implementation raises are not this module's to
                    # enumerate. Every one of them means the same thing here -- the guard cannot
                    # tell -- and the answer is to keep today's conservative behaviour. Logged
                    # with the type and message, because a permanently unreachable anchor is a
                    # silent return to a window an operator was told had gone.
                    log.warning(
                        "replay anchor unreadable; falling back to the creation-time watermark",
                        extra={
                            "extra_data": {
                                "bucket": self._bucket_name,
                                "error": f"{type(exc).__name__}: {exc}",
                            }
                        },
                    )
            if self._ledger_first_existed is not None:
                # The anchor is stamped on this fleet's clock and the creation time on the
                # broker's, so the same drift the watermark allows for applies to the
                # comparison. Inside that band the two happened together, which is a first run.
                replaced = self._ledger_first_existed < date_created - CLOCK_DRIFT_ALLOWANCE
        return replaced

    async def bind(self) -> "KvBucketLike":
        """open this guard's KV bucket, creating it when absent. Idempotent and async-safe.

        **A service calls this at startup, before it serves any artifact.** After a wipe the
        guard refuses every artifact issued before its bucket's creation time plus the refusal
        reach, and the bucket is created by whichever call opens it first. Binding at start moves
        that creation time before any artifact this process could issue or accept, so after a
        NATS wipe the watermark refuses only artifacts issued before the service started, or
        within the reach of its start. Left to the first :meth:`record_unique`, the bucket is
        created at first USE instead, and every artifact issued between the service starting and
        that first use -- including the one that triggered it -- is refused as a possible replay
        although it was issued after the service came up. After a restart that surfaces as a
        failed login or a refused tool call with nothing wrong but the order of two events.

        :meth:`record_unique` calls this too, so a guard nobody bound still works; it only pays
        that window. Every caller shares one open: concurrent calls wait on the first rather than
        opening the bucket again, and later calls return the handle without a round trip.

        **A wipe while the process keeps running is not re-bound here, and does not need to be
        for correctness.** The handle is kept for the process's life; the wrapper's self-heal
        recreates a vanished stream on the next operation through it, and :meth:`record_unique`
        reads the creation time fresh from the server after every fresh create, so a wipe at any
        moment can only make the check stricter. What it costs is availability: a bucket wiped
        under a running process is recreated at its next use, so that first artifact after the
        wipe is refused the same way.

        :return: the bound bucket handle
        :rtype: KvBucketLike
        :raises threetears.nats.KvError: when the bucket cannot be opened or created
        """
        if self._bucket is None:
            async with self._bucket_lock:
                if self._bucket is None:
                    # memory storage: a wipe is detected by record_unique's creation-time check,
                    # not survived. see the module docstring.
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
