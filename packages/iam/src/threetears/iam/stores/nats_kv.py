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

import hashlib
import json
import time
from collections.abc import Mapping
from datetime import timedelta
from typing import Any, Final

from threetears.nats.errors import KvError
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
        return _strip_internal(_decode(raw))

    async def get(self, key: str) -> Mapping[str, Any] | None:
        raw = await self._bucket.get(key=key)
        if raw is None:
            return None
        return _strip_internal(_decode(raw))


class NatsKvAttemptLimiter:
    """KV-backed :class:`~threetears.iam.stores.base.AttemptLimiter`, fixed-window.

    **The window is bucketed into the key, not left to the bucket TTL.** A JetStream KV TTL
    expires a key some interval after its last WRITE, which makes a TTL-only counter a
    sliding window: an attacker pacing attempts just inside the interval keeps one counter
    alive indefinitely and never gets a fresh window, while a legitimate user who fails a few
    times has their lockout silently extended by each attempt. Stamping the window ordinal
    into the key gives a true fixed window -- a new ordinal is simply a different key, which
    starts at zero -- and the bucket TTL then serves only to reap old ordinals.

    **Keys are hashed before they reach KV.** Callers key on an email address or a client IP,
    and neither belongs in a bucket listing an operator can dump. The hash is truncated
    because this is a partitioning key, not a security boundary: the value it protects is
    already known to whoever is being rate-limited.

    **Fails open.** A KV outage reports "not limited" rather than raising. That is a real
    trade -- a broker outage suspends rate limiting -- but the alternative is that the same
    outage locks every user out of every service, turning a degraded dependency into a total
    one. Rate limiting is a hardening layer, not a correctness gate. Every instance of it is
    logged, so the degradation is visible rather than assumed.
    """

    def __init__(
        self, bucket: NatsKvBucket, *, max_attempts: int = 5, window: timedelta = timedelta(minutes=15)
    ) -> None:
        self._bucket = bucket
        self._max_attempts = max_attempts
        self._window = window

    def _key(self, key: str) -> str:
        """Hash the caller's key and stamp the current window ordinal onto it."""
        digest = hashlib.sha256(key.lower().encode("utf-8")).hexdigest()[:32]
        ordinal = int(time.time()) // max(int(self._window.total_seconds()), 1)
        return f"attempt.{digest}.{ordinal}"

    def _verdict(self, count: int) -> AttemptWindow:
        limited = count >= self._max_attempts
        return AttemptWindow(count=count, limited=limited, retry_after=self._window if limited else None)

    async def record_failure(self, key: str) -> AttemptWindow:
        bucket_key = self._key(key)
        try:
            for _ in range(_CAS_ATTEMPTS):
                entry = await self._bucket.get_entry(key=bucket_key)
                if entry is None:
                    if await self._bucket.create(key=bucket_key, value=b"1") is not None:
                        return self._verdict(1)
                    continue
                raw, revision = entry
                count = _parse_count(raw) + 1
                updated = await self._bucket.update(key=bucket_key, value=str(count).encode("ascii"), revision=revision)
                if updated is not None:
                    return self._verdict(count)
        except KvError as exc:
            log.warning(
                "attempt-limiter record failed (KV unavailable); failing open",
                extra={"extra_data": {"error": str(exc)}},
            )
            return AttemptWindow(count=0, limited=False)
        # Losing every CAS attempt means heavy concurrent failures against ONE key, which is
        # itself the attack signal. Report limited rather than dropping the increment.
        log.warning("attempt-limiter contention: reporting limited without a recorded increment")
        return AttemptWindow(count=self._max_attempts, limited=True, retry_after=self._window)

    async def check(self, key: str) -> AttemptWindow:
        try:
            entry = await self._bucket.get_entry(key=self._key(key))
        except KvError as exc:
            log.warning(
                "attempt-limiter check failed (KV unavailable); failing open",
                extra={"extra_data": {"error": str(exc)}},
            )
            return AttemptWindow(count=0, limited=False)
        # get_entry rather than get: the increment path already needs the revision, so using
        # one accessor for both keeps the bucket surface this class depends on as small as
        # possible -- which is also what lets a caller's test double stay small.
        return self._verdict(_parse_count(entry[0]) if entry is not None else 0)

    async def clear(self, key: str) -> None:
        try:
            await self._bucket.delete(key=self._key(key))
        except KvError as exc:
            log.warning(
                "attempt-limiter clear failed (KV unavailable)",
                extra={"extra_data": {"error": str(exc)}},
            )


def _parse_count(raw: bytes) -> int:
    """Read a counter value, treating corruption as zero rather than raising."""
    try:
        return int(raw.decode("ascii"))
    except UnicodeDecodeError, ValueError:
        log.warning("discarding an unparseable attempt counter")
        return 0
