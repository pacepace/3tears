"""JetStream KV implementations of the storage Protocols.

**Expiry has two layers, and both are load-bearing.** The bucket's TTL reaps
storage: a store whose cleanup depends on someone remembering to sweep is a
store that accumulates live password-reset tickets for a year, and a bucket
opened with a TTL forgets on its own. But a bucket TTL is one number for every
entry in it, and the Protocol promises a per-call ``ttl`` -- "how long the
ticket stays redeemable". So each entry also carries its own absolute expiry,
checked on every read, which is what makes a ten-minute ticket in an
hour-long bucket actually stop working after ten minutes.

Without that second layer the per-call ``ttl`` was decoration: it was recorded
into the stored payload and read by nothing, so every entry was redeemable for
the whole bucket TTL. The in-memory double honoured it faithfully, which meant
the double enforced an expiry that production did not -- the failure mode a
test double is supposed to make impossible.

**The expiry is wall-clock, not monotonic.** Unlike the in-memory double,
these entries are read by a different process from the one that wrote them,
and monotonic clocks are not comparable across processes.

The two stores take a bucket already opened with the right TTL -- or come from
the :func:`state_store` / :func:`ticket_store` factories below, which open it
per call (``kv_bucket`` caches the handle, so that stays correct across a broker
reconnect).

**Redemption is a compare-and-swap claim, not a read-then-delete.** Two
concurrent redemptions of one ticket must produce exactly one winner. Reading
the value and then deleting it lets both callers read before either deletes,
which for a password-reset ticket means two parties both get to set the
password. So the delete is guarded by the revision the read observed, and only
the caller whose revision still matches wins. :meth:`NatsKvStateStore.get` is
the deliberate exception -- it does not consume, and is only correct where a
separate replay guard enforces single use.

**Counting is not here at all.** The
:class:`~threetears.iam.stores.base.AttemptLimiter` implementation lives in
:mod:`threetears.iam.stores.attempt_limiter`, because attempt counts are durable
state and stopped being KV in 0.43.0. Everything in THIS module really does hold
a bucket.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Mapping
from datetime import timedelta
from typing import TYPE_CHECKING, Any, Final

from threetears.observe import get_logger

from threetears.iam.stores.base import (
    TicketIssue,
    hash_ticket,
    new_ticket_secret,
)

if TYPE_CHECKING:
    # From the submodule, not the package: these are Protocols that `threetears.nats`
    # stopped re-exporting when its nats-py-backed surface went lazy. Annotation-only,
    # so the eager `kv` import here costs an L1 consumer nothing.
    from threetears.nats.kv import KvBucketLike, KvCapable

__all__ = [
    "NatsKvStateStore",
    "NatsKvTicketStore",
    "state_store",
    "ticket_store",
]

log = get_logger(__name__)


def _encode(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(dict(payload), separators=(",", ":")).encode("utf-8")


def _decode(raw: bytes) -> Mapping[str, Any] | None:
    """Decode a stored payload, treating corruption as absence.

    A value that will not parse is unusable either way; raising would turn it into a 500 on
    an authentication path, where the correct answer is simply "this ticket is not valid".
    """
    try:
        decoded = json.loads(raw.decode("utf-8"))
    except UnicodeDecodeError, json.JSONDecodeError:
        log.warning("discarding an unparseable KV payload")
        return None
    return decoded if isinstance(decoded, dict) else None


#: The absolute expiry stamped into every stored entry, unix seconds. Underscore-prefixed
#: and stripped on the way out, so a caller never sees it and cannot collide with it.
_EXPIRES_AT: Final[str] = "_expires_at"


def _strip_internal(payload: Mapping[str, Any] | None) -> Mapping[str, Any] | None:
    """Drop the bookkeeping keys this module adds, so callers see only what they stored."""
    if payload is None:
        return None
    return {name: value for name, value in payload.items() if name != _EXPIRES_AT}


def _live(payload: Mapping[str, Any] | None, now: float) -> Mapping[str, Any] | None:
    """The stored payload if it is still within its own TTL, else ``None``.

    An entry carrying no expiry at all is treated as live rather than as expired: the
    bucket TTL still bounds it, and defaulting the other way would make an unreadable
    stamp silently invalidate a valid ticket.
    """
    if payload is None:
        return None
    expires_at = payload.get(_EXPIRES_AT)
    if isinstance(expires_at, int | float) and not isinstance(expires_at, bool) and expires_at <= now:
        return None
    return payload


class NatsKvTicketStore:
    """KV-backed :class:`~threetears.iam.stores.base.SingleUseTicketStore`.

    Each ticket carries its own expiry, so ``ttl`` on :meth:`issue` is honoured exactly.
    The bucket's TTL is the storage reaper behind it and therefore still a ceiling: a
    ticket cannot outlive the bucket it sits in, so open the bucket with at least the
    longest ticket lifetime the caller issues.
    """

    def __init__(self, bucket: KvBucketLike, *, clock: Callable[[], float] = time.time) -> None:
        """
        :param bucket: a bucket already opened with a TTL at least as long as the longest
            ticket this store will issue.
        :ptype bucket: KvBucketLike
        :param clock: injectable wall clock, unix seconds. Wall rather than monotonic
            because the process that reads an entry is not the one that wrote it.
        :ptype clock: Callable[[], float]
        """
        self._bucket = bucket
        self._clock = clock

    async def issue(self, payload: Mapping[str, Any], *, ttl: timedelta) -> TicketIssue:
        secret = new_ticket_secret()
        hashed = hash_ticket(secret)
        # `create`, not `put`: a hash collision would silently overwrite a live ticket, and
        # while that is vanishingly unlikely at 256 bits, the SET-NX form costs nothing and
        # turns "impossible" into "detected".
        stored = {**dict(payload), _EXPIRES_AT: self._clock() + ttl.total_seconds()}
        if await self._bucket.create(key=hashed, value=_encode(stored)) is None:
            raise RuntimeError("ticket key already exists; refusing to overwrite a live ticket")
        return TicketIssue(secret=secret, hashed=hashed)

    async def redeem(self, secret: str) -> Mapping[str, Any] | None:
        key = hash_ticket(secret)
        entry = await self._bucket.get_entry(key=key)
        if entry is None:
            return None
        raw, revision = entry
        # Expiry is checked BEFORE the delete, so an expired ticket is refused without being
        # consumed -- the same order the Postgres store gets by putting the predicate inside
        # its DELETE. Consuming it first would make "expired" and "already redeemed"
        # indistinguishable in an audit trail, and would let anyone holding an expired secret
        # destroy the record of it.
        payload = _live(_decode(raw), self._clock())
        if payload is None:
            return None
        # Revision-guarded: whoever's delete matches the revision they read is the single
        # winner. A racing caller's delete fails and it correctly sees an unredeemable ticket.
        if not await self._bucket.delete(key=key, revision=revision):
            return None
        return _strip_internal(payload)


class NatsKvStateStore:
    """KV-backed :class:`~threetears.iam.stores.base.StateStore`.

    Each entry carries its own expiry, exactly as :class:`NatsKvTicketStore`'s tickets do;
    the bucket TTL is the reaper behind it.
    """

    def __init__(self, bucket: KvBucketLike, *, clock: Callable[[], float] = time.time) -> None:
        """
        :param bucket: a bucket already opened with a TTL at least as long as the longest
            entry this store will hold.
        :ptype bucket: KvBucketLike
        :param clock: injectable wall clock, unix seconds.
        :ptype clock: Callable[[], float]
        """
        self._bucket = bucket
        self._clock = clock

    async def put(self, key: str, payload: Mapping[str, Any], *, ttl: timedelta) -> None:
        stored = {**dict(payload), _EXPIRES_AT: self._clock() + ttl.total_seconds()}
        await self._bucket.put(key=key, value=_encode(stored))

    async def take(self, key: str) -> Mapping[str, Any] | None:
        entry = await self._bucket.get_entry(key=key)
        if entry is None:
            return None
        raw, revision = entry
        payload = _live(_decode(raw), self._clock())
        if payload is None:
            return None
        if not await self._bucket.delete(key=key, revision=revision):
            return None
        return _strip_internal(payload)

    async def get(self, key: str) -> Mapping[str, Any] | None:
        raw = await self._bucket.get(key=key)
        if raw is None:
            return None
        return _strip_internal(_live(_decode(raw), self._clock()))


async def state_store(nc: KvCapable, *, name: str, ttl: timedelta) -> NatsKvStateStore:
    """Open (or rebind) ``name`` and wrap it as a :class:`NatsKvStateStore`.

    Resolved per call rather than held: :meth:`~threetears.nats.kv.KvCapable.kv_bucket` caches
    the handle itself, so this costs nothing and stays correct across a broker reconnect --
    where a handle captured once at construction would not.

    **MEMORY storage, and that is a WITHDRAWAL rather than the original default.** 0.40.0
    made these factories request ``storage="file"``, on the reasoning that a memory-backed
    bucket "is not reliably readable by a second replica" and that this was the cause of an
    intermittent double login on a two-replica deployment.

    **The change was deployed, the buckets were deleted so they recreated file-backed, and
    the double login continued.** Verified on cobalt-dev 2026-09-15: every auth-flow bucket
    read ``file``, and a login still restarted once and then succeeded. The reasoning made a
    prediction and the prediction failed, so the reasoning is withdrawn rather than kept
    alongside a symptom it does not explain.

    What is true of JetStream is the ordinary thing: a stream exists once in the cluster,
    memory-backed or not, and any client on any node reaches it. Memory decides whether it
    survives a BROKER RESTART, not whether a second replica can read it. These buckets hold
    state with a ten-minute TTL, so a restart costs whoever is mid-login one retry.

    This is deliberately explicit rather than dropping the argument. ``kv_bucket`` defaults
    to memory, so omitting it would have the same effect and none of the history -- and the
    next person to meet the double login should find this paragraph rather than an absence
    that looks like nobody considered it.

    **The counter stores are NOT part of this withdrawal, and they are no longer buckets.** The
    reason ``WindowedCounter`` asked for file storage still holds -- a login lockout that resets
    to zero on every broker restart is not a lockout -- but the answer is L3, not a file-backed
    cache tier: it now keeps its counts in the coordination tables, with L2 as the fence in
    front. These two stores keep their buckets, and their ten-minute state is what a restart may
    cost one login.

    :param nc: the connected client.
    :ptype nc: KvCapable
    :param name: bucket suffix, namespace-prefixed by the client.
    :ptype name: str
    :param ttl: bucket TTL -- the storage reaper, and the ceiling on any per-call ``ttl``
        passed to :meth:`NatsKvStateStore.put`. The per-call value is what expires an
        individual entry; this is what eventually removes it.
    :ptype ttl: timedelta
    :return: the store.
    :rtype: NatsKvStateStore
    """
    return NatsKvStateStore(await nc.kv_bucket(name=name, ttl=ttl, storage="memory"))


async def ticket_store(nc: KvCapable, *, name: str, ttl: timedelta) -> NatsKvTicketStore:
    """Open (or rebind) ``name`` and wrap it as a :class:`NatsKvTicketStore`.

    Same per-call resolution as :func:`state_store`, and MEMORY storage for the same reason:
    the cross-replica argument for file storage was tested against the symptom it claimed to
    explain and did not fix it. See :func:`state_store` for what was tried and what actually
    holds.

    :param nc: the connected client.
    :ptype nc: KvCapable
    :param name: bucket suffix, namespace-prefixed by the client.
    :ptype name: str
    :param ttl: bucket TTL -- the storage reaper, and the ceiling on every ticket issued
        from the store. Each ticket's own ``ttl`` expires it; this eventually removes it.
    :ptype ttl: timedelta
    :return: the store.
    :rtype: NatsKvTicketStore
    """
    return NatsKvTicketStore(await nc.kv_bucket(name=name, ttl=ttl, storage="memory"))
