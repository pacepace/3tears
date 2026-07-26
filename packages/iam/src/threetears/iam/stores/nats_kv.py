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
reconnect). :class:`NatsKvAttemptLimiter` is the exception: it takes the client
and a bucket name, because the ``WindowedCounter`` underneath owns its own
bucket lifecycle. Either way, naming stays with the caller who knows the
deployment's namespace.

**Redemption is a compare-and-swap claim, not a read-then-delete.** Two
concurrent redemptions of one ticket must produce exactly one winner. Reading
the value and then deleting it lets both callers read before either deletes,
which for a password-reset ticket means two parties both get to set the
password. So the delete is guarded by the revision the read observed, and only
the caller whose revision still matches wins. :meth:`NatsKvStateStore.get` is
the deliberate exception -- it does not consume, and is only correct where a
separate replay guard enforces single use.

**Counting is not implemented here.** :class:`NatsKvAttemptLimiter` adapts
:class:`~threetears.core.coordination.WindowedCounter` to the
:class:`~threetears.iam.stores.base.AttemptLimiter` Protocol. There is one
windowed-counter implementation in the platform and it lives in
``threetears.core.coordination``, next to the other distributed security
primitives.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Mapping
from datetime import timedelta
from typing import Any, Final

from threetears.core.coordination import WindowedCounter, WindowState
from threetears.nats import KvCapable
from threetears.nats.kv import KvBucketLike
from threetears.observe import get_logger

from threetears.iam.stores.base import (
    AttemptWindow,
    TicketIssue,
    hash_ticket,
    new_ticket_secret,
)

__all__ = [
    "NatsKvAttemptLimiter",
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


class NatsKvAttemptLimiter:
    """:class:`~threetears.iam.stores.base.AttemptLimiter` over a
    :class:`~threetears.core.coordination.WindowedCounter`.

    The counting, the window, the CAS loop and the fail-open decision are all the counter's;
    this class only supplies the threshold and shapes the answer into an
    :class:`~threetears.iam.stores.base.AttemptWindow`. A second counting implementation
    would be a second set of window semantics to get wrong, which is exactly how a lockout
    ends up lasting a hundred milliseconds.

    **The window is anchored at the first failure, not at a wall-clock boundary.** Five
    failures buy a full window of lockout measured from the fifth attempt. An epoch-aligned
    window -- ``floor(now / window)`` -- looks equivalent and is not: every key's window
    rolls at the same instant, so an attacker who straddles a boundary gets ``2 x
    max_attempts`` back to back and a victim's lockout can expire almost immediately.

    **Fail-open is the caller's choice and defaults to closed.** It is defensible for a cheap
    edge throttle sitting in front of an authoritative check. It is not defensible for
    credential lockout, which has nothing behind it: there, a KV outage that reports "not
    limited" is an unlimited password-guessing window.
    """

    def __init__(
        self,
        nats_client: KvCapable,
        *,
        bucket_name: str,
        max_attempts: int = 5,
        window: timedelta = timedelta(minutes=15),
        fail_open: bool = False,
    ) -> None:
        """
        :param nats_client: the connected client; the counter opens its own bucket.
        :ptype nats_client: KvCapable
        :param bucket_name: bucket suffix, namespace-prefixed by the client. Give each
            protected surface its own, so unrelated counters never share a budget.
        :ptype bucket_name: str
        :param max_attempts: failures within one window before :attr:`AttemptWindow.limited`.
        :ptype max_attempts: int
        :param window: the window length, and the bucket TTL that reaps abandoned counters.
        :ptype window: timedelta
        :param fail_open: whether a KV transport failure reports "not limited" instead of
            raising. Defaults to ``False`` -- pass ``True`` only with an authoritative check
            behind this one.
        :ptype fail_open: bool
        """
        self._max_attempts = max_attempts
        self._window = window
        self._counter = WindowedCounter(
            nats_client,
            bucket_name=bucket_name,
            window_seconds=int(window.total_seconds()),
            fail_open=fail_open,
        )

    def _verdict(self, state: WindowState | None) -> AttemptWindow:
        if state is None:
            return AttemptWindow(count=0, limited=False)
        limited = state.count >= self._max_attempts
        if not limited:
            return AttemptWindow(count=state.count, limited=False)
        # Time actually remaining, not the window length: a caller surfacing `Retry-After`
        # should not tell a user to wait fifteen minutes when three are left.
        # Read through the counter's own clock, not `time.time()` -- two clocks in one
        # verdict is two answers, and the retry_after is the one a user is shown.
        elapsed = self._counter.clock() - state.window_start
        remaining = max(self._window.total_seconds() - elapsed, 0.0)
        return AttemptWindow(count=state.count, limited=True, retry_after=timedelta(seconds=remaining))

    async def record_failure(self, key: str) -> AttemptWindow:
        count = await self._counter.record_attempt(key)
        if count == 0:
            # fail-open: the counter swallowed a KvError and recorded nothing.
            return AttemptWindow(count=0, limited=False)
        return self._verdict(await self._counter.state(key))

    async def check(self, key: str) -> AttemptWindow:
        return self._verdict(await self._counter.state(key))

    async def clear(self, key: str) -> None:
        await self._counter.clear(key)


async def state_store(nc: KvCapable, *, name: str, ttl: timedelta) -> NatsKvStateStore:
    """Open (or rebind) ``name`` and wrap it as a :class:`NatsKvStateStore`.

    Resolved per call rather than held: :meth:`~threetears.nats.KvCapable.kv_bucket` caches
    the handle itself, so this costs nothing and stays correct across a broker reconnect --
    where a handle captured once at construction would not.

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
    return NatsKvStateStore(await nc.kv_bucket(name=name, ttl=ttl))


async def ticket_store(nc: KvCapable, *, name: str, ttl: timedelta) -> NatsKvTicketStore:
    """Open (or rebind) ``name`` and wrap it as a :class:`NatsKvTicketStore`.

    Same per-call resolution as :func:`state_store`, for the same reason.

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
    return NatsKvTicketStore(await nc.kv_bucket(name=name, ttl=ttl))
