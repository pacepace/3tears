"""3tears-observe: structured logging, tracing, and OpenTelemetry setup.

Provides three modules:

- ``threetears.observe.logging`` -- structured logging with generic context
  correlation and automatic call-site capture.
- ``threetears.observe.tracing`` -- ``@traced`` decorator that creates
  OpenTelemetry spans (zero-cost when OTel is not installed).
- ``threetears.observe.setup`` -- OpenTelemetry SDK bootstrap for host
  applications (TracerProvider, LoggerProvider, OTLP exporters).
"""

__version__ = "0.5.0"

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
