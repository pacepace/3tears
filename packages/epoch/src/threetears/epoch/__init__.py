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

# Version derived from pyproject.toml so the metadata is the single
# source of truth -- a future release that bumps pyproject without
# updating ``__init__.py`` can't drift the runtime ``__version__``.
# The except guard handles the rare case where the package isn't
# installed via importlib.metadata (e.g. running directly from a
# checked-out source tree without ``uv sync``); the fallback keeps
# imports working but reports ``unknown`` rather than crashing.
from importlib.metadata import PackageNotFoundError as _PackageNotFoundError
from importlib.metadata import version as _version

try:
    __version__ = _version("3tears-epoch")
except _PackageNotFoundError:  # pragma: no cover - dev fallback
    __version__ = "unknown"

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
