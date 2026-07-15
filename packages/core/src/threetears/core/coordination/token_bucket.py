"""TokenBucket — distributed token-bucket rate limiter over NATS JetStream KV.

generalizes 14-eng-ai-survey's Redis Lua-script token bucket
(``RedisRateLimiter``) and its own separate, non-atomic burst-limit
reimplementation of the same idea
(``RateLimitingCallback._check_burst_limit``, a plain ``GET`` then
``SET`` -- not atomic, a real race window between the two calls) into
one reusable, correct-by-construction primitive: tokens refill
continuously at a fixed rate up to a capacity ceiling; each claim
consumes tokens atomically via a CAS read-modify-write loop, the same
shape :class:`KVLease`/:class:`IdempotencyKeyStore` (this module's
siblings) already use.

usage::

    bucket = TokenBucket(
        nats_client, bucket_name="prod14_llm_throughput", refill_rate=2.0, capacity=10.0
    )
    outcome = await bucket.claim("global")
    if not outcome.claimed:
        # outcome.retry_after_seconds tells the caller how long until a
        # token frees up -- useful for an HTTP 429 Retry-After header
        ...

    # or block until a token is available (or 5s elapses):
    outcome = await bucket.claim("global", max_wait_seconds=5.0)

all KV envelope payloads flow through
:func:`threetears.core.serialization.serialize_to_json` /
:func:`deserialize_from_json`, matching :class:`KVLease`'s convention.
"""

from __future__ import annotations

import asyncio
import random
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any, Final

from threetears.core.serialization import deserialize_from_json, serialize_to_json
from threetears.observe import get_logger

if TYPE_CHECKING:
    from threetears.nats import NatsClient, NatsKvBucket

__all__ = [
    "TokenClaimResult",
    "TokenBucket",
    "TokenBucketConflict",
]

log = get_logger(__name__)

#: bucket-level KV TTL default -- an abandoned key (e.g. a per-user burst
#: bucket for a user who stopped sending requests) is forgotten after this
#: long, rather than growing the bucket's key set forever. the NEXT claim
#: against a forgotten key just starts fresh at full capacity (identical
#: observable behavior to a live key that happens to be at full capacity),
#: so this is a pure cleanup knob, not a correctness one.
_DEFAULT_KV_TTL: Final[timedelta] = timedelta(hours=1)

#: bounded CAS retry budget for claim()'s read-modify-write, when the CAS
#: update loses to a DIFFERENT concurrent claimer on the SAME key -- not
#: the same thing as "insufficient tokens" (see claim()'s docstring).
#: mirrors IdempotencyKeyStore's _CAS_MAX_RETRIES shape (this module's
#: sibling in packages/core/src/threetears/core/coordination/).
_CAS_MAX_RETRIES: Final[int] = 8

#: full-jitter backoff bound between CAS retries, seconds.
_CAS_RETRY_BACKOFF_SECONDS: Final[float] = 0.02


class TokenBucketConflict(RuntimeError):
    """raised when the CAS retry budget is exhausted claiming from a bucket.

    signals persistent write contention on ONE bucket key -- should be
    vanishingly rare given the bounded full-jitter backoff; exhausting
    the budget indicates unusually high concurrent claim pressure on a
    single key, rather than ordinary contention.
    """


@dataclass(frozen=True, slots=True)
class TokenClaimResult:
    """outcome of :meth:`TokenBucket.claim`.

    unlike :class:`KVLease`'s raise-on-denial convention, a rate-limit
    miss is an expected, common, per-request outcome rather than an
    exceptional one -- ``claim`` never raises for "not enough tokens
    right now," it always returns a result the caller inspects. this
    matches :class:`IdempotencyKeyStore`'s own claim()-returns-a-result
    convention more closely than :class:`KVLease`'s mutex semantics.

    :param claimed: whether tokens were successfully consumed
    :ptype claimed: bool
    :param tokens_remaining: bucket's token count immediately after this
        claim attempt (post-consume when claimed, current snapshot when not)
    :ptype tokens_remaining: float
    :param retry_after_seconds: estimated seconds until enough tokens will
        have refilled for a future claim of the same size to succeed;
        ``0.0`` when claimed
    :ptype retry_after_seconds: float
    """

    claimed: bool
    tokens_remaining: float
    retry_after_seconds: float


@dataclass
class _BucketState:
    """internal decoded KV value.

    :ivar tokens: token count as of ``last_refill``
    :ivar last_refill: timezone-aware datetime tokens were last computed at
    """

    tokens: float
    last_refill: datetime


def _encode_state(tokens: float, last_refill: datetime) -> bytes:
    """serialize bucket state to JSON bytes for KV storage.

    :param tokens: token count to store
    :ptype tokens: float
    :param last_refill: timezone-aware datetime this token count is as-of
    :ptype last_refill: datetime
    :return: JSON-encoded bytes suitable for a KV value
    :rtype: bytes
    """
    payload: dict[str, Any] = {"tokens": tokens, "last_refill": last_refill.isoformat()}
    return serialize_to_json(payload)


def _decode_state(data: bytes) -> _BucketState:
    """deserialize stored KV value bytes back to bucket state.

    :param data: bytes payload as returned from a KV bucket read
    :ptype data: bytes
    :return: decoded bucket state
    :rtype: _BucketState
    :raises ValueError: if payload is malformed or timestamp unparseable
    """
    raw = deserialize_from_json(data, field_types={})
    tokens = float(raw["tokens"])
    last_refill = datetime.fromisoformat(str(raw["last_refill"]))
    return _BucketState(tokens=tokens, last_refill=last_refill)


class TokenBucket:
    """distributed token-bucket rate limiter over one shared NATS JetStream KV bucket.

    one instance may serve many independent buckets, each keyed by a
    caller-supplied ``key`` (e.g. per-user, per-endpoint, or a single
    ``"global"`` key) -- ``refill_rate``/``capacity`` are fixed per
    :class:`TokenBucket` instance and apply uniformly to every key it
    serves; callers needing different rates for different keys construct
    separate instances (optionally sharing one underlying KV bucket_name
    only if their key namespaces cannot collide, though separate
    bucket_names is the safer default -- see IdempotencyKeyStore's own
    "pick a bucket dedicated to one domain" guidance).

    KV bucket binding is lazy (deferred to the first operation), matching
    :class:`KVLease`/:class:`IdempotencyKeyStore`'s construction style
    within this package.
    """

    def __init__(
        self,
        nats_client: "NatsClient",
        *,
        bucket_name: str,
        refill_rate: float,
        capacity: float,
        kv_ttl: timedelta | None = _DEFAULT_KV_TTL,
    ) -> None:
        """configure the bucket; defer KV bucket binding until first use.

        :param nats_client: connected canonical :class:`threetears.nats.NatsClient`;
            the store opens its KV bucket through :meth:`NatsClient.kv_bucket`
        :ptype nats_client: NatsClient
        :param bucket_name: KV bucket suffix; the wrapper prefixes it with the
            namespace. pick a bucket dedicated to one rate-limiting domain so
            unrelated keys never collide across surfaces
        :ptype bucket_name: str
        :param refill_rate: tokens added per second, continuously (not in
            discrete ticks) -- a claim 0.5 seconds after the last sees
            ``0.5 * refill_rate`` additional tokens available, matching the
            Redis Lua script's own fractional-elapsed-time math
        :ptype refill_rate: float
        :param capacity: maximum tokens one key can accumulate (the burst
            ceiling); refill never pushes a key's token count above this
        :ptype capacity: float
        :param kv_ttl: bucket-level KV TTL for inactive keys; defaults to
            :data:`_DEFAULT_KV_TTL`. ``None`` means keys never expire --
            rarely correct when the key space grows with something
            unbounded (e.g. per-user keys), since an abandoned key then
            lingers in the KV bucket forever
        :ptype kv_ttl: timedelta | None
        :return: none
        :rtype: None
        :raises ValueError: if refill_rate or capacity is not positive
        """
        if refill_rate <= 0:
            raise ValueError(f"refill_rate must be positive, got {refill_rate}")
        if capacity <= 0:
            raise ValueError(f"capacity must be positive, got {capacity}")
        self._client = nats_client
        self._bucket_name = bucket_name
        self._refill_rate = refill_rate
        self._capacity = capacity
        self._kv_ttl = kv_ttl
        self._bucket: "NatsKvBucket | None" = None
        self._bucket_lock = asyncio.Lock()

    @property
    def bucket_name(self) -> str:
        """the configured KV bucket suffix.

        :return: bucket name
        :rtype: str
        """
        return self._bucket_name

    async def claim(
        self, key: str = "default", *, tokens: float = 1.0, max_wait_seconds: float = 0.0
    ) -> TokenClaimResult:
        """attempt to claim ``tokens`` tokens from bucket ``key``.

        ``max_wait_seconds=0.0`` (the default): non-blocking, single
        attempt, returns immediately with ``claimed=False`` if
        insufficient tokens are available right now.

        ``max_wait_seconds>0.0``: blocks, retrying until claimed or the
        deadline elapses, matching :meth:`KVLease.acquire`'s own
        blocking-wait convention -- though here the wait sleeps toward
        the estimated refill time rather than a fixed poll interval,
        since the exact time a token becomes available is computable.

        never raises for "not enough tokens" (see :class:`TokenClaimResult`'s
        docstring for why) -- only raises on a genuine KV transport
        failure or exhausted CAS-contention retry budget.

        :param key: bucket key; independent token buckets never interact
        :ptype key: str
        :param tokens: number of tokens this claim consumes
        :ptype tokens: float
        :param max_wait_seconds: total seconds caller is willing to block;
            ``0.0`` disables blocking and forces a single fail-fast attempt
        :ptype max_wait_seconds: float
        :return: claim outcome; caller proceeds only when ``claimed`` is true
        :rtype: TokenClaimResult
        :raises ValueError: if ``tokens`` exceeds bucket capacity (can never
            be satisfied by any amount of waiting)
        :raises TokenBucketConflict: if the CAS retry budget is exhausted
        :raises threetears.nats.KvError: on a KV transport failure
        """
        if tokens > self._capacity:
            raise ValueError(f"cannot claim {tokens} tokens: exceeds bucket capacity {self._capacity}")
        bucket = await self._ensure_bucket()
        deadline = datetime.now(UTC) + timedelta(seconds=max_wait_seconds) if max_wait_seconds > 0 else None
        while True:
            result = await self._attempt(bucket, key, tokens)
            if result.claimed:
                return result
            if deadline is None:
                return result
            remaining = (deadline - datetime.now(UTC)).total_seconds()
            if remaining <= 0:
                return result
            await asyncio.sleep(max(0.0, min(result.retry_after_seconds, remaining)))

    async def _attempt(self, bucket: "NatsKvBucket", key: str, tokens: float) -> TokenClaimResult:
        """one refill-then-maybe-consume pass.

        retries internally, bounded by :data:`_CAS_MAX_RETRIES`, only
        when the CAS update loses to a DIFFERENT concurrent claimer
        touching the identical key this instant -- NOT when tokens are
        merely insufficient, which is a definitive (non-retryable)
        outcome this method returns directly.

        :param bucket: backing wrapper KV bucket (from :meth:`_ensure_bucket`)
        :ptype bucket: NatsKvBucket
        :param key: bucket key
        :ptype key: str
        :param tokens: number of tokens this claim consumes
        :ptype tokens: float
        :return: definitive claim outcome for this attempt
        :rtype: TokenClaimResult
        :raises TokenBucketConflict: if the CAS retry budget is exhausted
        """
        for attempt in range(_CAS_MAX_RETRIES):
            now = datetime.now(UTC)
            entry = await bucket.get_entry(key=key)
            if entry is None:
                current_tokens = self._capacity
                revision = None
            else:
                value, revision = entry
                state = _decode_state(value)
                elapsed = max(0.0, (now - state.last_refill).total_seconds())
                current_tokens = min(self._capacity, state.tokens + elapsed * self._refill_rate)

            if current_tokens < tokens:
                shortfall = tokens - current_tokens
                retry_after = shortfall / self._refill_rate
                return TokenClaimResult(claimed=False, tokens_remaining=current_tokens, retry_after_seconds=retry_after)

            new_tokens = current_tokens - tokens
            payload = _encode_state(new_tokens, now)
            if revision is None:
                # NatsKvBucket.create returns the new revision on success or
                # None on CAS conflict (another claimer created the key
                # between our get_entry miss and this create) -- retry.
                new_revision = await bucket.create(key=key, value=payload)
            else:
                # NatsKvBucket.update returns None on revision mismatch --
                # another claimer consumed from this key between our
                # get_entry and this update -- retry.
                new_revision = await bucket.update(key=key, value=payload, revision=revision)
            if new_revision is not None:
                return TokenClaimResult(claimed=True, tokens_remaining=new_tokens, retry_after_seconds=0.0)
            if attempt < _CAS_MAX_RETRIES - 1:
                backoff = random.uniform(0, _CAS_RETRY_BACKOFF_SECONDS)  # noqa: S311 - jitter, not security
                await asyncio.sleep(backoff)
        raise TokenBucketConflict(f"exhausted {_CAS_MAX_RETRIES} CAS retries claiming from bucket {key!r}")

    async def _ensure_bucket(self) -> "NatsKvBucket":
        """open (or bind) the TTL'd KV bucket once; async-safe lazy init."""
        if self._bucket is not None:
            return self._bucket
        async with self._bucket_lock:
            if self._bucket is None:
                self._bucket = await self._client.kv_bucket(
                    name=self._bucket_name,
                    ttl=self._kv_ttl,
                    storage="memory",
                    create_if_missing=True,
                    history=1,
                )
                log.info("TokenBucket bound bucket %s", self._bucket_name)
        return self._bucket
