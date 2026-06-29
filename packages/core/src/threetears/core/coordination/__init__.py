"""cross-pod coordination primitives backed by NATS JetStream KV.

public surface:

- :class:`KVLease` — distributed mutex factory with TTL and ownership tokens
- :class:`LeaseHandle` — per-acquire handle with refresh/release/async-with
- :class:`LeaseUnavailable` — raised by fail-fast acquire when key is held
- :class:`LeaseTimeout` — raised when acquire deadline elapses
- :class:`LeaseLost` — raised when ownership changes mid-operation
- :class:`ReplayGuard` — single-use nonce guard (shared, fail-closed) for replay protection
"""

from threetears.core.coordination.lease import (
    KVLease,
    LeaseHandle,
    LeaseLost,
    LeaseTimeout,
    LeaseUnavailable,
)
from threetears.core.coordination.replay_guard import ReplayGuard

__all__ = [
    "KVLease",
    "LeaseHandle",
    "LeaseLost",
    "LeaseTimeout",
    "LeaseUnavailable",
    "ReplayGuard",
]
