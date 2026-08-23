"""invalidation-listener enforcement domain -- cross-pod cache coherence wiring.

Every ``CollectionRegistry`` holding an L2-live collection must subscribe the cross-pod
invalidation stream once its NATS client is bound, and release that subscription in the
process's shutdown path. Publishing an invalidation is automatic on every write;
CONSUMING it is a separate step that gets forgotten, and the forgetting is silent -- the
pod serves its own stale L1 copy and nothing reports it.

Scoping the L2 keyspace per principal made this worse rather than better. Before, a peer
writing the same row refreshed a SHARED key underneath a forgetful pod; now each principal
has a private keyspace, nothing else ever writes those keys, and a row cached before a
sibling's write stays stale in both tiers for the life of the process.

Per-repo configuration goes through :class:`InvalidationListenerConfig`;
:func:`run_invalidation_listener_enforcement` is the pytest-friendly entry point.
"""

from threetears.enforcement.invalidation_listener.config import (
    InvalidationListenerConfig,
)
from threetears.enforcement.invalidation_listener.runner import (
    run_invalidation_listener_enforcement,
)
from threetears.enforcement.invalidation_listener.walkers import (
    START_NAMES,
    STOP_NAMES,
    count_l2_live_registries,
    find_starts_without_stops,
    find_unlistened_registries,
    started_registries,
    starts_without_stopping,
    unlistened_registries,
)

__all__ = [
    "START_NAMES",
    "STOP_NAMES",
    "InvalidationListenerConfig",
    "count_l2_live_registries",
    "find_starts_without_stops",
    "find_unlistened_registries",
    "run_invalidation_listener_enforcement",
    "started_registries",
    "starts_without_stopping",
    "unlistened_registries",
]
