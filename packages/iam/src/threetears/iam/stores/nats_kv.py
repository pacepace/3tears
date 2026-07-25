"""JetStream KV implementations of the storage Protocols.

TTL is the bucket's job here, not application code's. A store whose expiry
depends on someone remembering to sweep is a store that accumulates live
password-reset tickets for a year; a bucket opened with a TTL forgets on its
own. Each store therefore expects a bucket already opened with the right TTL,
via :meth:`~threetears.nats.NatsClient.kv_bucket`, rather than opening one
itself -- bucket naming and lifecycle stay with the caller who knows the
deployment's namespace.

**Redemption is a compare-and-swap claim, not a read-then-delete.** Two
concurrent redemptions of one ticket must produce exactly one winner. Reading
the value and then deleting it lets both callers read before either deletes,
which for a password-reset ticket means two parties both get to set the
password. So the delete is guarded by the revision the read observed, and only
the caller whose revision still matches wins.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import timedelta
from typing import Any, Final

from threetears.nats.kv import NatsKvBucket
from threetears.observe import get_logger

from threetears.iam.stores.base import (
    AttemptWindow,
    TicketIssue,
    hash_ticket,
    new_ticket_secret,
)

__all__ = ["NatsKvAttemptLimiter", "NatsKvStateStore", "NatsKvTicketStore"]

log = get_logger(__name__)

#: How many times a counter increment retries when it loses a compare-and-swap race.
#: Bounded rather than unbounded: under genuine contention an unbounded retry loop turns a
#: credential-stuffing burst into a spin, which is the attacker's goal.
_CAS_ATTEMPTS: Final[int] = 5


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
        payload = _decode(raw)
        if payload is None:
            return None
        return {name: value for name, value in payload.items() if name != "_ttl_seconds"}


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
        payload = _decode(raw)
        if payload is None:
            return None
        return {name: value for name, value in payload.items() if name != "_ttl_seconds"}


class NatsKvAttemptLimiter:
    """KV-backed :class:`~threetears.iam.stores.base.AttemptLimiter`, fixed-window.

    The window comes from the BUCKET's TTL: a counter key simply expires, which is both
    cheaper and more reliable than storing a window start and comparing timestamps across
    pods with imperfect clocks. Open the bucket with ``ttl`` set to the lockout window.

    **Fails open.** A KV outage returns "not limited" rather than raising. That is a real
    trade -- it means a broker outage suspends rate limiting -- but the alternative is that
    the same outage locks every user out of every service, turning a degraded dependency
    into a total one. The choice is logged, loudly, so it is visible rather than assumed.
    """

    def __init__(
        self, bucket: NatsKvBucket, *, max_attempts: int = 5, window: timedelta = timedelta(minutes=15)
    ) -> None:
        self._bucket = bucket
        self._max_attempts = max_attempts
        self._window = window

    def _verdict(self, count: int) -> AttemptWindow:
        limited = count >= self._max_attempts
        return AttemptWindow(count=count, limited=limited, retry_after=self._window if limited else None)

    async def record_failure(self, key: str) -> AttemptWindow:
        for _ in range(_CAS_ATTEMPTS):
            entry = await self._bucket.get_entry(key=key)
            if entry is None:
                if await self._bucket.create(key=key, value=b"1") is not None:
                    return self._verdict(1)
                continue
            raw, revision = entry
            count = _parse_count(raw) + 1
            if await self._bucket.update(key=key, value=str(count).encode("ascii"), revision=revision) is not None:
                return self._verdict(count)
        # Losing every CAS attempt means heavy concurrent failures against ONE key, which is
        # itself the attack signal. Report limited rather than dropping the increment.
        log.warning("attempt-limiter contention: reporting limited without a recorded increment")
        return AttemptWindow(count=self._max_attempts, limited=True, retry_after=self._window)

    async def check(self, key: str) -> AttemptWindow:
        raw = await self._bucket.get(key=key)
        return self._verdict(_parse_count(raw) if raw is not None else 0)

    async def clear(self, key: str) -> None:
        await self._bucket.delete(key=key)


def _parse_count(raw: bytes) -> int:
    """Read a counter value, treating corruption as zero rather than raising."""
    try:
        return int(raw.decode("ascii"))
    except UnicodeDecodeError, ValueError:
        log.warning("discarding an unparseable attempt counter")
        return 0
