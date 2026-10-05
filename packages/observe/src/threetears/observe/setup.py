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
from dataclasses import dataclass, field
from datetime import timedelta
from typing import TYPE_CHECKING, Protocol

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


class _ProviderFlush(Protocol):
    """a provider's ``force_flush``: every OpenTelemetry provider takes its budget as ``timeout_millis``."""

    def __call__(self, *, timeout_millis: int) -> object:
        """flush, using at most *timeout_millis* (which OpenTelemetry's batch processors ignore).

        :param timeout_millis: the milliseconds the flush may use
        :ptype timeout_millis: int
        :return: whether the provider reports everything flushed; read as a bool
        :rtype: object
        """
        ...


@dataclass(frozen=True)
class _SignalFlush:
    """one provider's flush, named by the signal it carries.

    :ivar signal: ``traces``, ``metrics`` or ``logs``
    :ivar flush: the provider's flush
    """

    signal: str
    flush: _ProviderFlush


class _BoundedFlush:
    """a sequence of provider flushes run on one daemon worker thread, which callers wait on for a bounded time.

    OpenTelemetry's batch processors ignore the timeout handed to ``force_flush``: they export
    until their queue is empty, however long the exporter's retries take (the SDK's own TODO cites
    open-telemetry/opentelemetry-python#4568). The only way to bound the call is not to make it on
    the caller's thread. The worker is a daemon, so one still inside an exporter when the caller
    gives up never holds the interpreter open at exit; it finishes, or dies with the process.
    """

    def __init__(self, flushes: tuple[_SignalFlush, ...], deadline: float) -> None:
        """
        builds the worker, not yet started.

        :param flushes: the flushes to run, in order
        :ptype flushes: tuple[_SignalFlush, ...]
        :param deadline: the ``time.monotonic()`` instant the flushes share as their budget
        :ptype deadline: float
        """
        from threetears.observe._otel_internals import FLUSH_THREAD_NAME

        self.signals = tuple(flush.signal for flush in flushes)
        self._flushes = flushes
        self._deadline = deadline
        self._lock = threading.Lock()
        self._running: str | None = None
        self._abandoned = False
        self._flushed = True
        self._worker = threading.Thread(target=self._work, name=FLUSH_THREAD_NAME, daemon=True)

    def start(self) -> None:
        """start the worker.

        :return: nothing
        :rtype: None
        """
        self._worker.start()

    def is_alive(self) -> bool:
        """whether the worker is still flushing.

        :return: ``True`` while a flush is still running
        :rtype: bool
        """
        return self._worker.is_alive()

    def join(self, seconds: float) -> None:
        """wait for the worker to finish, at most *seconds*.

        :param seconds: the longest to wait
        :ptype seconds: float
        :return: nothing
        :rtype: None
        """
        self._worker.join(seconds)

    def running_signal(self) -> str | None:
        """the signal the worker is flushing now, or flushed last.

        :return: the signal, or ``None`` before the first flush starts
        :rtype: str | None
        """
        with self._lock:
            return self._running

    def settle(self, timeout: timedelta) -> bool:
        """the caller's verdict once it has waited: whether everything flushed, reporting it if not finished.

        :param timeout: the timeout the caller waited under, for the report
        :ptype timeout: timedelta
        :return: whether the worker finished and every flush reported success
        :rtype: bool
        """
        with self._lock:
            finished = not self._worker.is_alive()
            self._abandoned = not finished
            running = self._running
            flushed = finished and self._flushed
        if not finished:
            not_started = self.signals[self.signals.index(running) + 1 :] if running is not None else self.signals
            logger.warning(
                "telemetry flush did not finish within its timeout",
                extra={
                    "extra_data": {
                        "signal": running,
                        "not_started": list(not_started),
                        "timeout_seconds": timeout.total_seconds(),
                    }
                },
            )
        return flushed

    def _work(self) -> None:
        """the worker: each flush in order, each handed what is left of the shared deadline.

        :return: nothing
        :rtype: None
        """
        for signal_flush in self._flushes:
            signal = signal_flush.signal
            with self._lock:
                self._running = signal
            remaining_millis = max(0, int((self._deadline - time.monotonic()) * 1000))
            raised = False
            try:
                completed = bool(signal_flush.flush(timeout_millis=remaining_millis))
            except Exception as exc:  # noqa: BLE001 -- prawduct:allow prawduct/broad-except -- a vendor exporter can raise anything, and one signal's failure must not stop the others flushing; logged, and reported as not flushed
                logger.warning(
                    "telemetry flush failed",
                    exc_info=True,
                    extra={"extra_data": {"signal": signal, "error_type": type(exc).__name__}},
                )
                completed = False
                raised = True
            with self._lock:
                self._flushed = self._flushed and completed
                # a raise was reported above as what it was; once the caller has given up it has
                # already said which flush held it
                report = not completed and not raised and not self._abandoned
            if report:
                # the provider itself returned False: its own flush ran out of the time it was handed
                logger.warning(
                    "telemetry flush did not finish within its timeout",
                    extra={"extra_data": {"signal": signal, "remaining_millis": remaining_millis}},
                )


def _seconds_until(deadline: float) -> float:
    """the seconds left before a ``time.monotonic()`` deadline, never negative.

    :param deadline: the deadline
    :ptype deadline: float
    :return: the seconds left
    :rtype: float
    """
    return max(0.0, deadline - time.monotonic())


class _SingleFlight:
    """the one slot a flush worker runs in: at most one flushes at a time, process-wide.

    a caller arriving while an earlier flush is still running waits for that one first, inside its
    own timeout, and only then starts its own. the earlier worker holds the SDK's export locks, so
    a second one beside it could only queue behind it, and stacking workers against a stuck
    exporter would grow a thread per call. this is a slot that each flush replaces, not a value
    built once.
    """

    def __init__(self) -> None:
        """
        an empty slot.
        """
        self._lock = threading.Lock()
        self._current: _BoundedFlush | None = None

    def flush(self, flushes: tuple[_SignalFlush, ...], timeout: timedelta) -> bool:
        """run *flushes* on one daemon worker once no earlier one is running; wait at most *timeout* in all.

        :param flushes: the flushes to run, in order
        :ptype flushes: tuple[_SignalFlush, ...]
        :param timeout: the longest this call waits, for an earlier flush and its own together
        :ptype timeout: timedelta
        :return: whether every flush finished within *timeout* and reported success
        :rtype: bool
        """
        if not flushes:
            return True
        deadline = time.monotonic() + timeout.total_seconds()
        while True:
            with self._lock:
                current = self._current
                earlier = current if current is not None and current.is_alive() else None
                if earlier is None:
                    own = _BoundedFlush(flushes, deadline)
                    own.start()
                    self._current = own
            if earlier is None:
                break
            earlier.join(_seconds_until(deadline))
            if earlier.is_alive():
                logger.warning(
                    "telemetry flush did not finish within its timeout",
                    extra={
                        "extra_data": {
                            "signal": earlier.running_signal(),
                            "held_by_an_earlier_flush": True,
                            "not_started": [flush.signal for flush in flushes],
                            "timeout_seconds": timeout.total_seconds(),
                        }
                    },
                )
                return False
        own.join(_seconds_until(deadline))
        return own.settle(timeout)


#: every flush this module runs, from :func:`force_flush_telemetry` and :func:`shutdown_telemetry` alike
_single_flight = _SingleFlight()


def force_flush_telemetry(timeout: timedelta = timedelta(seconds=2)) -> bool:
    """export every buffered span, metric and log record now, without shutting anything down.

    the public way to make telemetry observable at a known moment -- a test asserting on what its
    collector received, a job about to exit through a path that skips :func:`shutdown_telemetry`.
    flushes, in order, the tracer provider :func:`init_telemetry` installed, the global meter
    provider when it is an SDK one that can flush (3tears installs none itself, so this covers
    whatever the host configured), and the log export :func:`init_telemetry` started. logs go
    last so that a record logged while the others flush -- this function's own warning that a
    trace or metric flush failed, say -- is carried by the log flush.

    the guarantee: this returns within ``timeout``, plus the cost of starting one thread and
    writing one log line, whatever the providers do. OpenTelemetry's batch processors ignore the
    timeout they are given and export until their queue is empty, so the flushes run on a daemon
    worker thread and this waits for it no longer than ``timeout``; each flush is handed what is
    left of it. when the wait runs out this returns ``False`` and logs a WARNING naming the signal
    still flushing and those not yet started; the worker carries on in the background and, being a
    daemon, never holds the process open at exit.

    one flush runs at a time: a call made while an earlier call's flush is still running waits for
    that one first, inside its own ``timeout``, then flushes; it never starts a second worker
    beside a running one. if the earlier flush outlasts ``timeout``, this returns ``False`` with a
    WARNING naming the signal that flush is held on.

    a provider that is not configured is skipped and counts as flushed, so with nothing configured
    this returns ``True`` at once, without a thread. a flush that raises is logged at WARNING as a
    failure, naming the signal and the exception type, and counts as not flushed; it never stops
    the flushes after it. success is each provider's own report: OpenTelemetry's batch processors
    report a drained queue as flushed even when the exporter failed to deliver it, and the
    exporter logs that failure itself.

    :param timeout: the longest this call waits for every flush together
    :ptype timeout: timedelta
    :return: whether every configured provider finished its flush, and reported success, within ``timeout``
    :rtype: bool
    """
    from opentelemetry import metrics

    tracer_provider = _tracer_provider
    meter_flush = getattr(metrics.get_meter_provider(), "force_flush", None)
    log_export = _log_export
    candidates: tuple[tuple[str, _ProviderFlush | None], ...] = (
        ("traces", tracer_provider.force_flush if tracer_provider is not None else None),
        ("metrics", meter_flush if callable(meter_flush) else None),
        ("logs", log_export.force_flush if log_export is not None else None),
    )
    flushes = tuple(_SignalFlush(signal, flush) for signal, flush in candidates if flush is not None)
    return _single_flight.flush(flushes, timeout)


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
    the single flush worker and waited for at most two seconds, with a WARNING naming
    the signal when it does not finish. Each provider's own ``shutdown`` is then the
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
        _single_flight.flush((_SignalFlush("logs", log_export.force_flush),), _SHUTDOWN_FLUSH_TIMEOUT)
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
    # is logged, naming the signal: this process's last spans may be lost, and shutdown proceeds
    # regardless
    tracer_provider = _tracer_provider
    _single_flight.flush((_SignalFlush("traces", tracer_provider.force_flush),), _SHUTDOWN_FLUSH_TIMEOUT)

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
