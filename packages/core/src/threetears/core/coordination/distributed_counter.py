"""DistributedCounter — atomic increment/decrement counter over NATS JetStream KV.

generalizes "count occurrences of X, atomically, across pods" -- fixed-
window rate limiting (caller computes a time-bucketed key so a fresh
window naturally starts at zero once the previous window's key expires
via bucket TTL) and concurrent-in-flight tracking (caller increments on
start, decrements on end) both build on this same atomic-delta
primitive, rather than each hand-rolling a get/mutate/set loop.
generalizes 14-eng-ai-survey's Redis ``INCR``/``INCRBY`` + ``EXPIRE``
pattern (``RateLimitingCallback``'s request/token/concurrent limits,
``RateLimitMiddleware``'s per-endpoint limit) into a reusable cross-pod
primitive.

the primitive has no opinion about windowing or limit-checking -- only
atomic delta application. callers doing fixed-window rate limiting
supply a key that already encodes the window boundary (e.g.
``f"requests:{user_id}:{window_start}"``); callers tracking concurrent
in-flight work call :meth:`increment` on start and :meth:`decrement` on
completion against one stable key. checking the returned value against
a limit, and deciding whether to compensate with a decrement when
over, is the caller's responsibility -- this mirrors how
``RateLimitingCallback._check_concurrent_limit`` already works today
(increment first, decrement back out if the new value is over the
limit), which is race-free with nothing more than an atomic add.

usage::

    counter = DistributedCounter(
        nats_client, bucket_name="prod14_ratelimit_requests", ttl=timedelta(minutes=2)
    )
    new_value = await counter.increment(f"user-42:{window_start}")
    if new_value > limit:
        await counter.decrement(f"user-42:{window_start}")
        # reject this request

all KV envelope payloads flow through
:func:`threetears.core.serialization.serialize_to_json` /
:func:`deserialize_from_json`, matching :class:`KVLease`'s convention.
"""

from __future__ import annotations

import asyncio
import random
from datetime import timedelta
from typing import TYPE_CHECKING, Any, Final

from threetears.core.serialization import deserialize_from_json, serialize_to_json
from threetears.observe import get_logger

if TYPE_CHECKING:
    from threetears.nats import NatsClient, NatsKvBucket

__all__ = [
    "DistributedCounter",
    "DistributedCounterConflict",
]

log = get_logger(__name__)

#: bounded CAS retry budget for increment()/decrement()'s read-modify-write.
#: deliberately higher than IdempotencyKeyStore's 8 (this module's sibling):
#: that primitive's keys are mostly distinct per operation, so contention on
#: any ONE key is rare, whereas this primitive's whole point is many pods
#: hammering the SAME shared key (a fixed-window rate-limit counter, a
#: concurrent-in-flight gauge) -- a genuinely hotter access pattern. 30
#: matches 14-eng-ai-survey's own empirically-tuned constant for this exact
#: shape (IndexesData/SplitAssignmentsData, tuned against a live 20-way
#: concurrent integration test after 8 proved insufficient under real
#: multi-connection contention -- confirmed here too: a 25-connection
#: integration test against this primitive raised
#: DistributedCounterConflict on ~75% of runs at _CAS_MAX_RETRIES=8, zero
#: failures at 30, across repeated runs).
_CAS_MAX_RETRIES: Final[int] = 30

#: full-jitter backoff bound between CAS retries, seconds.
_CAS_RETRY_BACKOFF_SECONDS: Final[float] = 0.02


class DistributedCounterConflict(RuntimeError):
    """raised when the CAS retry budget is exhausted applying a delta.

    signals persistent write contention on ONE counter key -- should be
    vanishingly rare given the bounded full-jitter backoff; exhausting
    the budget indicates unusually high concurrent write pressure on a
    single key (many pods incrementing the identical key at once),
    rather than ordinary contention.
    """


def _encode_value(value: int) -> bytes:
    """serialize an integer counter value to JSON bytes for KV storage.

    :param value: counter value to encode
    :ptype value: int
    :return: JSON-encoded bytes suitable for a KV value
    :rtype: bytes
    """
    payload: dict[str, Any] = {"value": value}
    return serialize_to_json(payload)


def _decode_value(data: bytes) -> int:
    """deserialize stored KV value bytes back to an integer counter value.

    :param data: bytes payload as returned from a KV bucket read
    :ptype data: bytes
    :return: decoded counter value
    :rtype: int
    :raises ValueError: if payload is malformed
    """
    raw = deserialize_from_json(data, field_types={"value": int})
    return int(raw["value"])


class DistributedCounter:
    """atomic increment/decrement counter over one shared NATS JetStream KV bucket.

    bucket binding is lazy (deferred to the first operation), matching
    :class:`KVLease`/:class:`IdempotencyKeyStore`'s construction style
    within this package.
    """

    def __init__(self, nats_client: "NatsClient", *, bucket_name: str, ttl: timedelta | None = None) -> None:
        """configure the counter; defer bucket binding until first use.

        :param nats_client: connected canonical :class:`threetears.nats.NatsClient`;
            the counter opens its KV bucket through :meth:`NatsClient.kv_bucket`
        :ptype nats_client: NatsClient
        :param bucket_name: KV bucket suffix; dedicate one bucket per counting
            domain (e.g. HTTP-request windows vs. LLM-token windows) so
            unrelated keys never collide across surfaces, and so each domain
            can pick its own TTL
        :ptype bucket_name: str
        :param ttl: how long an untouched key is remembered before the bucket
            expires it; ``None`` means keys never expire -- rarely correct
            for a fixed-window counter, since an unbounded bucket grows
            forever as new windows mint new keys. callers doing fixed-window
            rate limiting should set this to roughly the window size;
            callers tracking concurrent in-flight work should set this to a
            generous safety margin (a decrement that never runs -- crash,
            forgotten call -- must not leak the slot forever)
        :ptype ttl: timedelta | None
        :return: none
        :rtype: None
        """
        self._client = nats_client
        self._bucket_name = bucket_name
        self._ttl = ttl
        self._bucket: "NatsKvBucket | None" = None
        self._bucket_lock = asyncio.Lock()

    @property
    def bucket_name(self) -> str:
        """the configured bucket suffix.

        :return: bucket name
        :rtype: str
        """
        return self._bucket_name

    async def increment(self, key: str, *, delta: int = 1) -> int:
        """atomically add ``delta`` to ``key``'s counter, return the new value.

        creates the key at ``delta`` if absent (the first increment on a
        fresh window or counter). backed by a CAS read-modify-write loop,
        bounded by a fixed retry budget with full-jitter backoff -- not a
        wall-clock deadline, since an atomic add either succeeds this
        attempt or loses to a concurrent writer and must retry immediately;
        unlike a blocking-wait-for-availability primitive (:class:`KVLease`)
        where genuinely waiting for a state change makes sense, there is
        nothing to wait FOR here.

        :param key: counter key
        :ptype key: str
        :param delta: amount to add; negative values decrement
        :ptype delta: int
        :return: new counter value after applying delta
        :rtype: int
        :raises DistributedCounterConflict: if the CAS retry budget is exhausted
        :raises threetears.nats.KvError: on a KV transport failure
        """
        return await self._apply_delta(key, delta)

    async def decrement(self, key: str, *, delta: int = 1) -> int:
        """convenience for ``increment(key, delta=-delta)``.

        the primitive does not clamp at zero -- a counter tracking
        concurrent in-flight work that decrements more times than it
        incremented (a caller bug) will go negative rather than silently
        floor at zero, surfacing the bug instead of masking it.

        :param key: counter key
        :ptype key: str
        :param delta: amount to subtract (a positive number)
        :ptype delta: int
        :return: new counter value after applying delta
        :rtype: int
        :raises DistributedCounterConflict: if the CAS retry budget is exhausted
        :raises threetears.nats.KvError: on a KV transport failure
        """
        return await self._apply_delta(key, -delta)

    async def get(self, key: str) -> int:
        """read ``key``'s current counter value.

        :param key: counter key
        :ptype key: str
        :return: current value, or 0 if the key has never been touched (or expired)
        :rtype: int
        :raises threetears.nats.KvError: on a KV transport failure
        """
        bucket = await self._ensure_bucket()
        value = await bucket.get(key=key)
        return _decode_value(value) if value is not None else 0

    async def _apply_delta(self, key: str, delta: int) -> int:
        """CAS read-modify-write ``key``, retrying under contention.

        :param key: counter key
        :ptype key: str
        :param delta: signed amount to add
        :ptype delta: int
        :return: new counter value after applying delta
        :rtype: int
        :raises DistributedCounterConflict: if the CAS retry budget is exhausted
        """
        bucket = await self._ensure_bucket()
        for attempt in range(_CAS_MAX_RETRIES):
            entry = await bucket.get_entry(key=key)
            if entry is None:
                new_value = delta
                # NatsKvBucket.create returns the new revision on success or
                # None on CAS conflict (another writer created the key
                # between our get_entry miss and this create) -- retry.
                revision = await bucket.create(key=key, value=_encode_value(new_value))
                if revision is not None:
                    return new_value
            else:
                current_value, revision = entry
                new_value = _decode_value(current_value) + delta
                # NatsKvBucket.update returns None on revision mismatch --
                # another writer applied a delta between our get_entry and
                # this update -- retry.
                new_revision = await bucket.update(key=key, value=_encode_value(new_value), revision=revision)
                if new_revision is not None:
                    return new_value
            if attempt < _CAS_MAX_RETRIES - 1:
                backoff = random.uniform(0, _CAS_RETRY_BACKOFF_SECONDS)  # noqa: S311 - jitter, not security
                await asyncio.sleep(backoff)
        raise DistributedCounterConflict(f"exhausted {_CAS_MAX_RETRIES} CAS retries applying delta to counter {key!r}")

    async def _ensure_bucket(self) -> "NatsKvBucket":
        """open (or bind) the TTL'd KV bucket once; async-safe lazy init."""
        if self._bucket is not None:
            return self._bucket
        async with self._bucket_lock:
            if self._bucket is None:
                self._bucket = await self._client.kv_bucket(
                    name=self._bucket_name,
                    ttl=self._ttl,
                    storage="memory",
                    create_if_missing=True,
                    history=1,
                )
                log.info("DistributedCounter bound bucket %s", self._bucket_name)
        return self._bucket
