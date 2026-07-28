"""OpenTelemetry SDK bootstrap for host applications.

Provides a single entry point for wiring up OTel tracing and log export.
All configuration comes from a ``TelemetryConfig`` dataclass -- host apps
build this from their own settings.  Standard ``OTEL_*`` env vars are
explicitly suppressed to prevent ambient config leaking in.

When disabled or the collector endpoint is unreachable, the API runs
with NoOp providers (zero overhead).
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from threetears.observe.logging import get_logger

if TYPE_CHECKING:
    from opentelemetry.sdk._logs import LoggerProvider
    from opentelemetry.sdk.trace import TracerProvider

__all__ = [
    "TelemetryConfig",
    "init_telemetry",
    "reset_telemetry",
    "shutdown_telemetry",
]

logger = get_logger(__name__)


@dataclass(frozen=True)
class TelemetryConfig:
    """Configuration for OpenTelemetry setup.

    Host applications build this from their own settings and pass it to
    ``init_telemetry()``.  All fields have sensible defaults for local
    development.
    """

    enabled: bool = False
    endpoint: str = "http://localhost:4317"
    service_name: str = "threetears"
    service_version: str = "0.1.0"
    sample_rate: float = 1.0
    export_timeout_seconds: int = 10
    loki_endpoint: str | None = None
    suppressed_env_vars: tuple[str, ...] = field(
        default=(
            "OTEL_EXPORTER_OTLP_ENDPOINT",
            "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT",
            "OTEL_SERVICE_NAME",
            "OTEL_TRACES_SAMPLER",
            "OTEL_TRACES_SAMPLER_ARG",
            "OTEL_RESOURCE_ATTRIBUTES",
        )
    )


# ---------------------------------------------------------------------------
# Module-level state
# ---------------------------------------------------------------------------

_tracer_provider: TracerProvider | None = None
_log_provider: LoggerProvider | None = None
_log_handler: logging.Handler | None = None
_shutdown_called: bool = False


# ---------------------------------------------------------------------------
# Call-site enriching handler (bridges ThreeTearsLogger attrs to OTel)
# ---------------------------------------------------------------------------


class _CallSiteEnrichingHandler(logging.Handler):
    """Logging handler that enriches OTel LogRecords with call-site attributes.

    The standard OTel LoggingHandler maps Python's ``pathname``/``funcName``/
    ``lineno`` to ``code.filepath``/``code.function``/``code.lineno``.
    ``ThreeTearsLogger`` sets enriched ``call_site_*`` attributes on the Python
    LogRecord.  This handler patches those onto the LogRecord's standard fields
    *before* the OTel handler processes them, so the downstream collector
    receives the enriched values.
    """

    def __init__(self, otel_handler: logging.Handler) -> None:
        super().__init__()
        self._otel_handler = otel_handler

    def emit(self, record: logging.LogRecord) -> None:
        """Enrich the LogRecord then forward to the OTel handler."""
        call_site_file = getattr(record, "call_site_file", None)
        if call_site_file:
            record.pathname = call_site_file
        call_site_class = getattr(record, "call_site_class", None)
        call_site_func = getattr(record, "call_site_func", None)
        if call_site_class and call_site_func:
            record.funcName = f"{call_site_class}.{call_site_func}"
        elif call_site_func:
            record.funcName = call_site_func
        call_site_line = getattr(record, "call_site_line", None)
        if call_site_line:
            record.lineno = call_site_line
        self._otel_handler.emit(record)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def init_telemetry(config: TelemetryConfig) -> bool:
    """Initialize OpenTelemetry tracing and (optionally) log export.

    Sets up a ``TracerProvider`` with OTLP gRPC span export and ratio-based
    sampling.  Standard ``OTEL_*`` env vars are suppressed to prevent ambient
    configuration from leaking in.

    If ``config.loki_endpoint`` is set, also initializes OTel log export
    (Python logging -> OTLP -> Loki).

    Safe to call multiple times (resets the SDK's once-guard internally).

    :param config: telemetry configuration built by the host application.
    :returns: True if tracing was successfully initialized, False if disabled.
    """
    global _tracer_provider, _shutdown_called  # noqa: PLW0603

    if not config.enabled:
        logger.info("OpenTelemetry tracing disabled")
        return False

    # Suppress ambient OTEL_* env vars
    for var in config.suppressed_env_vars:
        os.environ.pop(var, None)

    from opentelemetry import trace
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.sdk.trace.sampling import TraceIdRatioBased

    resource = Resource.create(
        {
            "service.name": config.service_name,
            "service.version": config.service_version,
        }
    )

    sampler = TraceIdRatioBased(config.sample_rate)
    provider = TracerProvider(resource=resource, sampler=sampler)

    exporter = OTLPSpanExporter(
        endpoint=config.endpoint,
        timeout=config.export_timeout_seconds,
        insecure=True,
    )

    provider.add_span_processor(BatchSpanProcessor(exporter))

    # Reset the once-only flag so we can (re-)set the provider.
    # Needed on OTel SDK >=1.39 where set_tracer_provider is guarded by
    # _TRACER_PROVIDER_SET_ONCE which only allows a single set.
    try:
        trace._TRACER_PROVIDER_SET_ONCE._done = False  # type: ignore[attr-defined, unused-ignore]
    except AttributeError:
        # The private guard this reaches into was renamed or removed by an SDK upgrade. The
        # set_tracer_provider below then keeps whatever provider was installed first, so tracing
        # quietly stops reaching our exporter -- the one failure here that has to be loud.
        logger.warning(
            "could not reset the OTel set-once guard; set_tracer_provider may be a no-op",
            extra={"extra_data": {"guard": "trace._TRACER_PROVIDER_SET_ONCE._done"}},
        )
    trace.set_tracer_provider(provider)
    _tracer_provider = provider
    _shutdown_called = False

    logger.info(
        "OpenTelemetry tracing initialized",
        extra={
            "extra_data": {
                "endpoint": config.endpoint,
                "service_name": config.service_name,
                "sample_rate": config.sample_rate,
            }
        },
    )

    # Log export (Python logging -> OTLP -> Loki)
    if config.loki_endpoint:
        _init_log_export(config, resource)

    return True


def _init_log_export(config: TelemetryConfig, resource: object) -> None:
    """Initialize OTel log export (Python logging -> OTLP -> Loki).

    Attaches a LoggingHandler to the root logger so every log record is exported
    as an OTel log record with trace context (trace_id, span_id) attached.
    The handler is wrapped to enrich OTel records with call-site info.
    """
    global _log_provider, _log_handler  # noqa: PLW0603

    from opentelemetry._logs import set_logger_provider
    from opentelemetry.exporter.otlp.proto.http._log_exporter import OTLPLogExporter
    from opentelemetry.sdk._logs import LoggerProvider
    from opentelemetry.sdk._logs import LoggingHandler
    from opentelemetry.sdk._logs.export import BatchLogRecordProcessor

    log_provider = LoggerProvider(resource=resource)  # type: ignore[arg-type]
    loki_otlp_url = f"http://{config.loki_endpoint}/otlp/v1/logs"
    log_exporter = OTLPLogExporter(endpoint=loki_otlp_url)
    log_provider.add_log_record_processor(BatchLogRecordProcessor(log_exporter))
    set_logger_provider(log_provider)

    otel_handler = LoggingHandler(level=logging.DEBUG, logger_provider=log_provider)
    handler = _CallSiteEnrichingHandler(otel_handler)
    logging.root.addHandler(handler)

    _log_provider = log_provider
    _log_handler = handler

    logger.info(
        "OTel log export initialized",
        extra={"extra_data": {"endpoint": config.loki_endpoint}},
    )


def shutdown_telemetry() -> None:
    """Flush pending spans and log records, then shut down OTel providers.

    Removes the log handler from the root logger, force-flushes and shuts down
    the ``LoggerProvider``, then force-flushes and shuts down the
    ``TracerProvider``.  After shutdown, resets the global tracer provider to
    ``NoOpTracerProvider`` and clears the SDK's once-guard so that
    ``init_telemetry()`` can be called again.

    Safe to call multiple times -- second and subsequent calls are no-ops.
    """
    global _shutdown_called, _tracer_provider, _log_provider, _log_handler  # noqa: PLW0603

    if _shutdown_called:
        return

    _shutdown_called = True

    # Shut down log provider first (it may emit logs during trace shutdown)
    if _log_handler is not None:
        logging.root.removeHandler(_log_handler)
        _log_handler = None

    if _log_provider is not None:
        # Broad on purpose -- a vendor exporter can raise anything on teardown, and neither
        # failure may stop the shutdown that follows it or prevent ``init_telemetry()``
        # being called again. NOT silent, though: an earlier version swallowed both with
        # `pass`, reasoning that the OTel handler had just been detached from the root
        # logger so there was nowhere left to report. That reasoning was wrong. Detaching
        # one handler does not disable the root logger; every other handler a host app
        # installed (console, file) is still attached and still receiving. All the silence
        # bought was an unexportable telemetry backend failing invisibly at every shutdown.
        try:
            _log_provider.force_flush(timeout_millis=2000)
        except Exception:  # noqa: BLE001 -- prawduct:allow prawduct/broad-except -- a vendor exporter can raise anything on teardown and must not stop the shutdown that follows; logged, not silenced
            logger.warning("telemetry shutdown: log provider flush failed", exc_info=True)
        try:
            _log_provider.shutdown()
        except Exception:  # noqa: BLE001 -- prawduct:allow prawduct/broad-except -- a vendor exporter can raise anything on teardown and must not prevent init_telemetry() being callable again; logged, not silenced
            logger.warning("telemetry shutdown: log provider shutdown failed", exc_info=True)
        _log_provider = None

    if _tracer_provider is None:
        return

    from opentelemetry import trace
    from opentelemetry.trace import NoOpTracerProvider

    try:
        _tracer_provider.force_flush(timeout_millis=2000)
    except Exception as exc:  # noqa: BLE001 -- shutdown continues regardless
        # The buffered spans did not make it to the exporter, so this process's last traces are
        # gone. Shutdown still proceeds; the operator just needs to know the tail is missing.
        logger.warning(
            "OTel force_flush failed; buffered spans were dropped",
            extra={"extra_data": {"error": str(exc)}},
        )

    try:
        _tracer_provider.shutdown()
    except Exception as exc:  # noqa: BLE001 -- shutdown continues regardless
        logger.warning(
            "OTel provider shutdown failed; exporter resources may not be released",
            extra={"extra_data": {"error": str(exc)}},
        )

    # Reset the global provider so new init_telemetry calls work
    try:
        trace._TRACER_PROVIDER_SET_ONCE._done = False  # type: ignore[attr-defined, unused-ignore]
    except AttributeError:
        # Same private guard as init_telemetry: without the reset a later init_telemetry cannot
        # install its provider, so tracing never comes back after this shutdown.
        logger.warning(
            "could not reset the OTel set-once guard; a later init_telemetry may be a no-op",
            extra={"extra_data": {"guard": "trace._TRACER_PROVIDER_SET_ONCE._done"}},
        )
    trace.set_tracer_provider(NoOpTracerProvider())
    _tracer_provider = None

    logger.info("OpenTelemetry shut down")


def reset_telemetry() -> None:
    """Shut down providers and reset all module-level state for test isolation.

    Calls ``shutdown_telemetry()`` then clears the ``_shutdown_called`` flag
    so that ``init_telemetry()`` can reinitialize from scratch.  Intended for
    test fixtures -- not for production use.
    """
    global _shutdown_called  # noqa: PLW0603
    shutdown_telemetry()
    _shutdown_called = False
