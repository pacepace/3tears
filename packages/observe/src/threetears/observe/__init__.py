"""3tears-observe: structured logging, tracing, and OpenTelemetry setup.

Provides three modules:

- ``threetears.observe.logging`` -- structured logging with generic context
  correlation and automatic call-site capture.
- ``threetears.observe.tracing`` -- ``@traced`` decorator that creates
  OpenTelemetry spans (zero-cost when OTel is not installed).
- ``threetears.observe.setup`` -- OpenTelemetry SDK bootstrap for host
  applications (TracerProvider, LoggerProvider, OTLP exporters).
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
from threetears.observe.health import HealthCheck, HealthServer
from threetears.observe.logging import (
    ContextFormatter,
    ThreeTearsLogger,
    clear_context,
    configure_logging,
    configure_third_party_logging,
    get_context,
    get_logger,
    set_context,
)
from threetears.observe.resilience import retry_with_backoff
from threetears.observe.tracing import traced

__all__ = [
    "ContextFormatter",
    "HealthCheck",
    "HealthServer",
    "ThreeTearsLogger",
    "clear_context",
    "configure_logging",
    "configure_third_party_logging",
    "get_context",
    "get_logger",
    "set_context",
    "retry_with_backoff",
    "spawn_background",
    "traced",
]
