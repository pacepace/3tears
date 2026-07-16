"""cross-pod coordination primitives backed by NATS JetStream KV.

public surface:

- :class:`KVLease` — distributed mutex factory with TTL and ownership tokens
- :class:`LeaseHandle` — per-acquire handle with refresh/release/async-with
- :class:`LeaseUnavailable` — raised by fail-fast acquire when key is held
- :class:`LeaseTimeout` — raised when acquire deadline elapses
- :class:`LeaseLost` — raised when ownership changes mid-operation
- :class:`ReplayGuard` — single-use nonce guard (shared, fail-closed) for replay protection
- :class:`RevocationGuard` — timestamped revocation entries (shared, fail-closed), for the
  "denylist everything that started before this moment" shape a bare presence test can't express
- :class:`WindowedCounter` — generic windowed attempt counter/rate-limiter (fail-open or
  fail-closed, caller's choice), for the "how many times in the last N seconds" shape a bare
  presence test can't express
- :class:`IdempotencyKeyStore` — claim-once-with-TTL primitive, stores operation result/error
- :class:`IdempotencyRecord` — one idempotency key's current state
- :class:`ClaimResult` — outcome of :meth:`IdempotencyKeyStore.claim`
- :class:`IdempotencyKeyNotFound` — raised by complete()/fail() on an unclaimed key
- :class:`IdempotencyConflict` — raised when a complete()/fail() CAS retry budget is exhausted
- :class:`TokenBucket` — distributed token-bucket rate limiter
- :class:`TokenClaimResult` — outcome of :meth:`TokenBucket.claim`
- :class:`TokenBucketConflict` — raised when a claim()'s CAS retry budget is exhausted
- :class:`DistributedCounter` — atomic increment/decrement counter, for fixed-window
  rate limiting and concurrent-in-flight tracking
- :class:`DistributedCounterConflict` — raised when an increment()/decrement()'s CAS
  retry budget is exhausted
"""

from threetears.core.coordination.distributed_counter import (
    DistributedCounter,
    DistributedCounterConflict,
)
from threetears.core.coordination.idempotency import (
    ClaimResult,
    IdempotencyConflict,
    IdempotencyKeyNotFound,
    IdempotencyKeyStore,
    IdempotencyRecord,
)
from threetears.core.coordination.lease import (
    KVLease,
    LeaseHandle,
    LeaseLost,
    LeaseTimeout,
    LeaseUnavailable,
)
from threetears.core.coordination.replay_guard import ReplayGuard, RevocationGuard
from threetears.core.coordination.token_bucket import (
    TokenBucket,
    TokenBucketConflict,
    TokenClaimResult,
)
from threetears.core.coordination.windowed_counter import WindowedCounter

__all__ = [
    "ClaimResult",
    "DistributedCounter",
    "DistributedCounterConflict",
    "IdempotencyConflict",
    "IdempotencyKeyNotFound",
    "IdempotencyKeyStore",
    "IdempotencyRecord",
    "KVLease",
    "LeaseHandle",
    "LeaseLost",
    "LeaseTimeout",
    "LeaseUnavailable",
    "ReplayGuard",
    "RevocationGuard",
    "TokenBucket",
    "TokenBucketConflict",
    "TokenClaimResult",
    "WindowedCounter",
]
