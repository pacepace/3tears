"""3tears-observe: structured logging, tracing, and OpenTelemetry setup.

The names a consumer imports from here are exactly those in ``__all__``; each
module's own docstring says what it is for.
"""

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
    __version__ = _version("3tears-observe")
except _PackageNotFoundError:  # pragma: no cover - dev fallback
    __version__ = "unknown"

from threetears.observe.background import spawn_background
from threetears.observe.build_once import BuildOnce
from threetears.observe.erasure import ANONYMIZED_MARKER
from threetears.observe.periodic import PeriodicTask, TickResult
from threetears.observe.health import HealthCheck, HealthServer, HealthTier
from threetears.observe.inflight import InflightRequestsGauge
from threetears.observe.logging import (
    ContextFormatter,
    ThreeTearsLogger,
    clear_context,
    NOISY_LIBRARY_LOGGERS,
    configure_logging,
    configure_third_party_logging,
    get_context,
    get_logger,
    representative_exception,
    set_context,
)
from threetears.observe.metrics import counter, gauge, histogram, metered
from threetears.observe.resilience import retry_until_done, retry_with_backoff
from threetears.observe.tracing import set_span_attribute, traced

__all__ = [
    "ANONYMIZED_MARKER",
    "BuildOnce",
    "ContextFormatter",
    "HealthCheck",
    "HealthServer",
    "HealthTier",
    "InflightRequestsGauge",
    "PeriodicTask",
    "ThreeTearsLogger",
    "TickResult",
    "clear_context",
    "NOISY_LIBRARY_LOGGERS",
    "configure_logging",
    "configure_third_party_logging",
    "counter",
    "gauge",
    "get_context",
    "get_logger",
    "histogram",
    "metered",
    "representative_exception",
    "set_context",
    "set_span_attribute",
    "retry_until_done",
    "retry_with_backoff",
    "spawn_background",
    "traced",
]
