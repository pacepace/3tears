"""3tears-epoch: cross-pod config-epoch coherence.

generation-stamped configuration epochs with NATS broadcast (push) and
per-message epoch echo (pull-on-stale) to keep in-memory caches
coherent across pods on admin writes.

provides three modules:

- :mod:`threetears.epoch.wire` -- :class:`EpochBumpMessage` typed wire
  envelope
- :mod:`threetears.epoch.client` -- :class:`EpochClient` publish-side
  bump + current-read against ``config_epochs``
- :mod:`threetears.epoch.listener` -- :class:`EpochListener`
  subscribe-side dispatcher with monotonic dedupe + echo helper
"""

from __future__ import annotations

__version__ = "0.7.0"

from threetears.epoch.client import EpochClient, PoolLike
from threetears.epoch.listener import BumpCallback, EpochListener
from threetears.epoch.wire import EpochBumpMessage

__all__ = [
    "BumpCallback",
    "EpochBumpMessage",
    "EpochClient",
    "EpochListener",
    "PoolLike",
]
