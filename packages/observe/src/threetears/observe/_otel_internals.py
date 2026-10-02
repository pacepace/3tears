"""the one owner of every OpenTelemetry private attribute and private module this package touches.

OpenTelemetry's API (``opentelemetry-api`` >= 1.39) guards ``trace.set_tracer_provider`` with a
module-level set-once flag, ``trace._TRACER_PROVIDER_SET_ONCE``, and offers no public way to clear
it. A second ``set_tracer_provider`` is then silently ignored, so tracing keeps reaching whichever
provider was installed first. :func:`threetears.observe.init_telemetry` must be able to install a
provider again after :func:`threetears.observe.shutdown_telemetry` (a host that re-initializes, and
every test fixture that does), and shutdown must hand the slot back for that.

So this module is the only place in ``threetears.observe`` that reads or writes an OpenTelemetry
name with a leading underscore. ``setup.py`` calls :func:`allow_tracer_provider_reset`, never the
attribute, so an OpenTelemetry release that renames the guard is reported here, by name, instead of
as an ``AttributeError`` from wherever it was reached for.

The surface: ``opentelemetry.trace._TRACER_PROVIDER_SET_ONCE`` (an ``opentelemetry.util._once.Once``)
and its ``_done`` flag.

**The logs API** (owner ruling 3, 2026-10-01). OpenTelemetry ships its logs API and SDK only as
private modules -- ``opentelemetry._logs``, ``opentelemetry.sdk._logs``,
``opentelemetry.sdk._logs.export`` -- and the OTLP HTTP log exporter only as
``opentelemetry.exporter.otlp.proto.http._log_exporter``. There is no public spelling of any of
them. So this module is also the only place that imports them: :func:`start_log_export` builds
the provider, batch processor, exporter and ``LoggingHandler`` that ``setup.py`` installs, and
:class:`OtelLogExport` is what ``setup.py`` holds to flush and shut it down. When an upgrade moves
those modules, importing this module fails naming the one that moved, and
``tests/test_otel_internals.py`` -- the guard -- fails with it; when it keeps the names but
changes how a record travels to the exporter, the guard's round trip fails.

Consumers never import these modules: a host app gets log export through
:func:`threetears.observe.init_telemetry` with ``loki_endpoint`` set.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Protocol, cast

from opentelemetry import trace
from opentelemetry._logs import set_logger_provider
from opentelemetry.exporter.otlp.proto.http._log_exporter import OTLPLogExporter
from opentelemetry.sdk._logs import LoggerProvider, LoggingHandler
from opentelemetry.sdk._logs.export import BatchLogRecordProcessor
from opentelemetry.sdk._logs.export import LogRecordExporter as SdkLogRecordExporter
from opentelemetry.sdk.resources import Resource

__all__ = ["LogRecordExporter", "OtelLogExport", "allow_tracer_provider_reset", "start_log_export"]


class LogRecordExporter(Protocol):
    """what :class:`BatchLogRecordProcessor` drives: the OpenTelemetry SDK's exporter surface.

    Declared here because the SDK's own ``LogRecordExporter`` lives in a private module.
    """

    def export(self, batch: Sequence[object], /) -> object:
        """send one batch of readable log records.

        :param batch: the records
        :ptype batch: Sequence[object]
        :return: the SDK's export result
        :rtype: object
        """
        ...

    def force_flush(self, timeout_millis: int = ...) -> bool:
        """flush anything buffered.

        :param timeout_millis: how long to wait
        :ptype timeout_millis: int
        :return: whether it flushed in time
        :rtype: bool
        """
        ...

    def shutdown(self) -> None:
        """release the exporter's resources."""
        ...


class OtelLogExport:
    """one running log export: the handler to attach, and the provider behind it.

    :ivar handler: the OpenTelemetry ``LoggingHandler``; attach it (or a wrapper) to a logger
    """

    def __init__(self, provider: LoggerProvider, handler: logging.Handler) -> None:
        """
        holds a started export.

        :param provider: the SDK logger provider the handler emits into
        :ptype provider: LoggerProvider
        :param handler: the handler bound to that provider
        :ptype handler: logging.Handler
        """
        self.provider = provider
        self.handler = handler

    def force_flush(self, timeout_millis: int) -> None:
        """export every buffered record now.

        :param timeout_millis: how long to wait for the exporter
        :ptype timeout_millis: int
        """
        self.provider.force_flush(timeout_millis=timeout_millis)

    def shutdown(self) -> None:
        """stop the processor and the exporter; records emitted after this are dropped."""
        self.provider.shutdown()


def allow_tracer_provider_reset() -> bool:
    """clear OpenTelemetry's set-once guard so the next ``set_tracer_provider`` takes effect.

    :return: ``True`` when the guard was cleared; ``False`` when this OpenTelemetry release no
        longer carries it in the shape above, in which case a later ``set_tracer_provider`` may be
        ignored and the caller must say so
    :rtype: bool
    """
    cleared = True
    try:
        trace._TRACER_PROVIDER_SET_ONCE._done = False  # type: ignore[attr-defined, unused-ignore]
    except AttributeError:
        # NOSILENT: the False return is the report; the caller logs what it costs at its own site
        cleared = False
    return cleared


def start_log_export(
    resource: Resource,
    otlp_logs_url: str,
    *,
    exporter: LogRecordExporter | None = None,
) -> OtelLogExport:
    """build and install OpenTelemetry log export: provider, batch processor, exporter, handler.

    The provider is also installed as OpenTelemetry's global logger provider, so the logs API
    reaches it too.

    :param resource: the service resource every record carries
    :ptype resource: Resource
    :param otlp_logs_url: the OTLP HTTP logs endpoint, for the default exporter
    :ptype otlp_logs_url: str
    :param exporter: the exporter records are batched into; the OTLP HTTP exporter for
        *otlp_logs_url* when ``None``, which is what production passes
    :ptype exporter: LogRecordExporter | None
    :return: the running export
    :rtype: OtelLogExport
    """
    provider = LoggerProvider(resource=resource)
    chosen = exporter if exporter is not None else OTLPLogExporter(endpoint=otlp_logs_url)
    # the SDK's exporter base is nominal; anything meeting the protocol above is what it drives
    provider.add_log_record_processor(BatchLogRecordProcessor(cast(SdkLogRecordExporter, chosen)))
    set_logger_provider(provider)
    handler = LoggingHandler(level=logging.DEBUG, logger_provider=provider)
    return OtelLogExport(provider, handler)
