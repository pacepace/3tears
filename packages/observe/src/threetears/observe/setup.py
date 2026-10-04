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
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import timedelta
from typing import TYPE_CHECKING

from threetears.observe.logging import get_logger

if TYPE_CHECKING:
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider

    from threetears.observe._otel_internals import OtelLogExport

__all__ = [
    "TelemetryConfig",
    "force_flush_telemetry",
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
_log_export: OtelLogExport | None = None
_log_handler: logging.Handler | None = None
_shutdown_called: bool = False


# ---------------------------------------------------------------------------
# Call-site enriching handler (bridges ThreeTearsLogger attrs to OTel)
# ---------------------------------------------------------------------------


class _CallSiteEnrichingHandler(logging.Handler):
    """Logging handler that enriches OTel LogRecords with call-site attributes.

    The OTel LoggingHandler (opentelemetry-instrumentation-logging's, built with
    ``log_code_attributes=True``) maps Python's ``pathname``/``funcName``/
    ``lineno`` to ``code.file.path``/``code.function.name``/``code.line.number``.
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
        # handle, not emit: the OTel handler's own filters must apply, and the one that keeps
        # OpenTelemetry's own records out of the export is what stops a failing export feeding itself
        self._otel_handler.handle(record)


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

    from threetears.observe._otel_internals import allow_tracer_provider_reset

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

    # Reset the once-only flag so we can (re-)set the provider: on OTel >=1.39
    # set_tracer_provider is guarded so it only takes effect once. See _otel_internals.
    if not allow_tracer_provider_reset():
        # The private guard was renamed or removed by an SDK upgrade. The set_tracer_provider
        # below then keeps whatever provider was installed first, so tracing quietly stops
        # reaching our exporter -- the one failure here that has to be loud.
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


def _init_log_export(config: TelemetryConfig, resource: Resource) -> None:
    """Initialize OTel log export (Python logging -> OTLP -> Loki).

    Attaches a LoggingHandler to the root logger so every log record is exported
    as an OTel log record with trace context (trace_id, span_id) attached.
    The handler is wrapped to enrich OTel records with call-site info.
    """
    global _log_export, _log_handler  # noqa: PLW0603

    # OpenTelemetry's logs API exists only as private modules; _otel_internals is their one owner.
    from threetears.observe._otel_internals import start_log_export

    log_export = start_log_export(resource, f"http://{config.loki_endpoint}/otlp/v1/logs")
    handler = _CallSiteEnrichingHandler(log_export.handler)
    logging.root.addHandler(handler)

    _log_export = log_export
    _log_handler = handler

    logger.info(
        "OTel log export initialized",
        extra={"extra_data": {"endpoint": config.loki_endpoint}},
    )


class _BoundedFlush:
    """a sequence of provider flushes run on a daemon worker thread, which the caller waits on for a bounded time.

    OpenTelemetry's batch processors ignore the timeout handed to ``force_flush``: they export
    until their queue is empty, however long the exporter's retries take (the SDK's own TODO cites
    open-telemetry/opentelemetry-python#4568). The only way to bound the call is not to make it on
    the caller's thread. The worker is a daemon, so one still inside an exporter when the caller
    gives up never holds the interpreter open at exit; it finishes, or dies with the process.
    """

    def __init__(self, flushes: list[tuple[str, Callable[[int], bool | None]]], timeout: timedelta) -> None:
        """
        holds the flushes to run and the budget they share.

        :param flushes: each signal's name and the flush that takes the milliseconds left to it
        :ptype flushes: list[tuple[str, Callable[[int], bool | None]]]
        :param timeout: the longest the caller waits for all of them together
        :ptype timeout: timedelta
        """
        self._flushes = flushes
        self._timeout = timeout
        self._deadline = 0.0
        self._lock = threading.Lock()
        self._running: str | None = None
        self._abandoned = False
        self._flushed = True

    def run(self) -> bool:
        """run every flush on a daemon worker and wait for it at most the timeout.

        :return: whether every flush finished within the timeout and reported success
        :rtype: bool
        """
        if not self._flushes:
            return True
        self._deadline = time.monotonic() + self._timeout.total_seconds()
        worker = threading.Thread(target=self._work, name="threetears-telemetry-flush", daemon=True)
        worker.start()
        worker.join(self._timeout.total_seconds())
        with self._lock:
            finished = not worker.is_alive()
            self._abandoned = not finished
            running = self._running
            flushed = finished and self._flushed
        if not finished:
            names = [signal for signal, _ in self._flushes]
            not_started = names[names.index(running) + 1 :] if running is not None else names
            logger.warning(
                "telemetry flush did not finish within its timeout",
                extra={
                    "extra_data": {
                        "signal": running,
                        "not_started": not_started,
                        "timeout_seconds": self._timeout.total_seconds(),
                    }
                },
            )
        return flushed

    def _work(self) -> None:
        """the worker: each flush in order, each handed what is left of the shared deadline.

        :return: nothing
        :rtype: None
        """
        for signal, flush in self._flushes:
            with self._lock:
                self._running = signal
            remaining_millis = max(0, int((self._deadline - time.monotonic()) * 1000))
            try:
                completed = bool(flush(remaining_millis))
            except Exception:  # noqa: BLE001 -- prawduct:allow prawduct/broad-except -- a vendor exporter can raise anything, and one signal's failure must not stop the others flushing; logged, and reported as not flushed
                logger.warning("telemetry flush failed", exc_info=True, extra={"extra_data": {"signal": signal}})
                completed = False
            with self._lock:
                self._flushed = self._flushed and completed
                # once the caller has given up it has already said which flush held it
                report = not completed and not self._abandoned
            if report:
                logger.warning(
                    "telemetry flush did not finish within its timeout",
                    extra={"extra_data": {"signal": signal, "timeout_seconds": self._timeout.total_seconds()}},
                )


def force_flush_telemetry(timeout: timedelta = timedelta(seconds=2)) -> bool:
    """export every buffered span, metric and log record now, without shutting anything down.

    the public way to make telemetry observable at a known moment -- a test asserting on what its
    collector received, a job about to exit through a path that skips :func:`shutdown_telemetry`.
    flushes, in order, the tracer provider :func:`init_telemetry` installed, the global meter
    provider when it is an SDK one that can flush (3tears installs none itself, so this covers
    whatever the host configured), and the log export :func:`init_telemetry` started. traces go
    first because flushing them can emit log records the log flush then carries.

    the guarantee: this returns within ``timeout``, plus the cost of starting one thread and
    writing one log line, whatever the providers do. OpenTelemetry's batch processors ignore the
    timeout they are given and export until their queue is empty, so the flushes run on a daemon
    worker thread and this waits for it no longer than ``timeout``; each flush is handed what is
    left of it. when the wait runs out this returns ``False`` and logs a WARNING naming the signal
    still flushing and those not yet started; the worker carries on in the background and, being a
    daemon, never holds the process open at exit.

    a provider that is not configured is skipped and counts as flushed, so with nothing configured
    this returns ``True`` at once, without a thread. a flush that raises is logged at WARNING and
    counts as not flushed; it never stops the flushes after it. success is each provider's own
    report: OpenTelemetry's batch processors report a drained queue as flushed even when the
    exporter failed to deliver it, and the exporter logs that failure itself.

    :param timeout: the longest this call waits for every flush together
    :ptype timeout: timedelta
    :return: whether every configured provider finished its flush, and reported success, within ``timeout``
    :rtype: bool
    """
    from opentelemetry import metrics

    flushes: list[tuple[str, Callable[[int], bool | None]]] = []
    if _tracer_provider is not None:
        tracer_provider = _tracer_provider
        flushes.append(("traces", lambda millis: tracer_provider.force_flush(timeout_millis=millis)))
    meter_flush = getattr(metrics.get_meter_provider(), "force_flush", None)
    if callable(meter_flush):
        flushes.append(("metrics", lambda millis: meter_flush(timeout_millis=millis)))
    if _log_export is not None:
        log_export = _log_export
        flushes.append(("logs", lambda millis: log_export.force_flush(timeout_millis=millis)))
    return _BoundedFlush(flushes, timeout).run()


#: how long shutdown waits for each provider's flush before shutting it down regardless
_SHUTDOWN_FLUSH_TIMEOUT = timedelta(seconds=2)


def shutdown_telemetry() -> None:
    """Flush pending spans and log records, then shut down OTel providers.

    Removes the log handler from the root logger, force-flushes and shuts down
    the ``LoggerProvider``, then force-flushes and shuts down the
    ``TracerProvider``.  After shutdown, resets the global tracer provider to
    ``NoOpTracerProvider`` and clears the SDK's once-guard so that
    ``init_telemetry()`` can be called again.

    Each flush is bounded the way :func:`force_flush_telemetry` bounds it: run on
    a daemon worker and waited for at most two seconds, with a WARNING naming the
    signal when it does not finish. Each provider's own ``shutdown`` is then the
    SDK's, which bounds itself: its batch processor stops accepting records, waits
    at most thirty seconds for its export worker, and then shuts the exporter down,
    which ends any retry backoff still in progress.

    Safe to call multiple times -- second and subsequent calls are no-ops.
    """
    global _shutdown_called, _tracer_provider, _log_export, _log_handler  # noqa: PLW0603

    if _shutdown_called:
        return

    _shutdown_called = True

    # Shut down log provider first (it may emit logs during trace shutdown)
    if _log_handler is not None:
        logging.root.removeHandler(_log_handler)
        _log_handler = None

    if _log_export is not None:
        # Bounded like force_flush_telemetry, and for the same reason: the SDK's flush ignores
        # its timeout, and an unreachable collector would otherwise hold shutdown in the
        # exporter's retries. A flush that raises or runs out of time is logged, never silent:
        # detaching the OTel handler above does not disable the root logger, and every other
        # handler a host app installed (console, file) still receives the warning.
        log_export = _log_export
        _BoundedFlush(
            [("logs", lambda millis: log_export.force_flush(timeout_millis=millis))], _SHUTDOWN_FLUSH_TIMEOUT
        ).run()
        # Broad on purpose -- a vendor exporter can raise anything on teardown, and the failure
        # may not prevent ``init_telemetry()`` being called again; logged, for the reason above.
        try:
            _log_export.shutdown()
        except Exception:  # noqa: BLE001 -- prawduct:allow prawduct/broad-except -- a vendor exporter can raise anything on teardown and must not prevent init_telemetry() being callable again; logged, not silenced
            logger.warning("telemetry shutdown: log provider shutdown failed", exc_info=True)
        _log_export = None

    if _tracer_provider is None:
        return

    from opentelemetry import trace
    from opentelemetry.trace import NoOpTracerProvider

    from threetears.observe._otel_internals import allow_tracer_provider_reset

    # bounded for the same reason as the log flush above. a flush that raises or runs out of time
    # is logged by _BoundedFlush, naming the signal: this process's last spans may be lost, and
    # shutdown proceeds regardless
    tracer_provider = _tracer_provider
    _BoundedFlush(
        [("traces", lambda millis: tracer_provider.force_flush(timeout_millis=millis))], _SHUTDOWN_FLUSH_TIMEOUT
    ).run()

    try:
        _tracer_provider.shutdown()
    except Exception as exc:  # noqa: BLE001 -- shutdown continues regardless
        logger.warning(
            "OTel provider shutdown failed; exporter resources may not be released",
            extra={"extra_data": {"error": str(exc)}},
        )

    # Reset the global provider so new init_telemetry calls work
    if not allow_tracer_provider_reset():
        # Same guard as init_telemetry: without the reset a later init_telemetry cannot
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
