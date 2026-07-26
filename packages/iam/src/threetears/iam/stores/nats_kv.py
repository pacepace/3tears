"""JetStream KV implementations of the storage Protocols.

TTL is the bucket's job here, not application code's. A store whose expiry
depends on someone remembering to sweep is a store that accumulates live
password-reset tickets for a year; a bucket opened with a TTL forgets on its
own. The two stores take a bucket already opened with the right TTL -- or come from
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
from collections.abc import Mapping
from datetime import timedelta
from typing import Any, Final

from threetears.core.coordination import WindowedCounter, WindowState
from threetears.nats import NatsClient
from threetears.nats.kv import NatsKvBucket
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


def _strip_internal(payload: Mapping[str, Any] | None) -> Mapping[str, Any] | None:
    """Drop the bookkeeping keys this module adds, so callers see only what they stored."""
    if payload is None:
        return None
    return {name: value for name, value in payload.items() if name != "_ttl_seconds"}


class NatsKvTicketStore:
    """KV-backed :class:`~threetears.iam.stores.base.SingleUseTicketStore`.

    The bucket's TTL bounds every ticket, so ``ttl`` on :meth:`issue` cannot extend past it.
    Open the bucket with the longest ticket lifetime the caller issues.
    """

    def __init__(self, bucket: NatsKvBucket) -> None:
        self._bucket = bucket

    async def issue(self, payload: Mapping[str, Any], *, ttl: timedelta) -> TicketIssue:
        secret = new_ticket_secret()
        hashed = hash_ticket(secret)
        # `create`, not `put`: a hash collision would silently overwrite a live ticket, and
        # while that is vanishingly unlikely at 256 bits, the SET-NX form costs nothing and
        # turns "impossible" into "detected".
        stored = {**dict(payload), "_ttl_seconds": int(ttl.total_seconds())}
        if await self._bucket.create(key=hashed, value=_encode(stored)) is None:
            raise RuntimeError("ticket key already exists; refusing to overwrite a live ticket")
        return TicketIssue(secret=secret, hashed=hashed)

    async def redeem(self, secret: str) -> Mapping[str, Any] | None:
        key = hash_ticket(secret)
        entry = await self._bucket.get_entry(key=key)
        if entry is None:
            return None
        raw, revision = entry
        # Revision-guarded: whoever's delete matches the revision they read is the single
        # winner. A racing caller's delete fails and it correctly sees an unredeemable ticket.
        if not await self._bucket.delete(key=key, revision=revision):
            return None
        return _strip_internal(_decode(raw))


class NatsKvStateStore:
    """KV-backed :class:`~threetears.iam.stores.base.StateStore`."""

    def __init__(self, bucket: NatsKvBucket) -> None:
        self._bucket = bucket

    async def put(self, key: str, payload: Mapping[str, Any], *, ttl: timedelta) -> None:
        stored = {**dict(payload), "_ttl_seconds": int(ttl.total_seconds())}
        await self._bucket.put(key=key, value=_encode(stored))

    async def take(self, key: str) -> Mapping[str, Any] | None:
        entry = await self._bucket.get_entry(key=key)
        if entry is None:
            return None
        raw, revision = entry
        if not await self._bucket.delete(key=key, revision=revision):
            return None
        return _strip_internal(_decode(raw))

    async def get(self, key: str) -> Mapping[str, Any] | None:
        raw = await self._bucket.get(key=key)
        if raw is None:
            return None
        return _strip_internal(_decode(raw))


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
        nats_client: NatsClient,
        *,
        bucket_name: str,
        max_attempts: int = 5,
        window: timedelta = timedelta(minutes=15),
        fail_open: bool = False,
    ) -> None:
        """
        :param nats_client: the connected client; the counter opens its own bucket.
        :ptype nats_client: NatsClient
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


async def state_store(nc: NatsClient, *, name: str, ttl: timedelta) -> NatsKvStateStore:
    """Open (or rebind) ``name`` and wrap it as a :class:`NatsKvStateStore`.

    Resolved per call rather than held: :meth:`~threetears.nats.NatsClient.kv_bucket` caches
    the handle itself, so this costs nothing and stays correct across a broker reconnect --
    where a handle captured once at construction would not.

    :param nc: the connected client.
    :ptype nc: NatsClient
    :param name: bucket suffix, namespace-prefixed by the client.
    :ptype name: str
    :param ttl: bucket TTL. This, not the per-call ``ttl`` on :meth:`NatsKvStateStore.put`,
        is what actually expires entries.
    :ptype ttl: timedelta
    :return: the store.
    :rtype: NatsKvStateStore
    """
    return NatsKvStateStore(await nc.kv_bucket(name=name, ttl=ttl))


async def ticket_store(nc: NatsClient, *, name: str, ttl: timedelta) -> NatsKvTicketStore:
    """Open (or rebind) ``name`` and wrap it as a :class:`NatsKvTicketStore`.

    Same per-call resolution as :func:`state_store`, for the same reason.

    :param nc: the connected client.
    :ptype nc: NatsClient
    :param name: bucket suffix, namespace-prefixed by the client.
    :ptype name: str
    :param ttl: bucket TTL, which bounds every ticket issued from the store.
    :ptype ttl: timedelta
    :return: the store.
    :rtype: NatsKvTicketStore
    """
    return NatsKvTicketStore(await nc.kv_bucket(name=name, ttl=ttl))
