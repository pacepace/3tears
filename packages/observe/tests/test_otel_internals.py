"""the OpenTelemetry private surface ``threetears.observe`` depends on still works in the installed release.

Owner ruling 3, 2026-10-01: OpenTelemetry's logs API exists only as ``opentelemetry._logs`` and
``opentelemetry.sdk._logs`` (and the OTLP HTTP log exporter only under ``..._log_exporter``), so
every import of them is confined to :mod:`threetears.observe._otel_internals`. This is that
module's own test. An OpenTelemetry release that moves or renames one of those modules fails here
at import, naming the module; one that keeps the names but changes how a record travels from a
python logger to an exporter fails the round trip below.

The handler is ``opentelemetry-instrumentation-logging``'s (owner ruling, 2026-10-01): the SDK's
own ``LoggingHandler`` is deprecated in its favour and warns on construction. The replacement adds
the ``code.*`` attributes only when asked to, and the call-site enrichment in ``setup.py`` exists to
set exactly those, so the round trip below also pins that they still arrive.
"""

from __future__ import annotations

import logging
import warnings
from collections.abc import Sequence

from opentelemetry.instrumentation.logging.handler import LoggingHandler
from opentelemetry.sdk.resources import Resource

from threetears.observe._otel_internals import OtelLogExport, start_log_export


# parity-exempt: duck-typed stand-in for the OpenTelemetry SDK's LogRecordExporter, whose module (opentelemetry.sdk._logs.export) is private and may not be bound from a test; it implements export, force_flush and shutdown, the whole surface BatchLogRecordProcessor drives
class FakeLogRecordExporter:
    """collects every exported record body; reports success to the processor."""

    def __init__(self) -> None:
        self.bodies: list[object] = []
        self.attributes: list[dict[str, object]] = []
        self.shut_down = False

    def export(self, batch: Sequence[object]) -> None:
        """record each exported record's body.

        :param batch: the SDK's readable log records
        :ptype batch: Sequence[object]
        """
        for record in batch:
            log_record = getattr(record, "log_record")
            self.bodies.append(log_record.body)
            self.attributes.append(dict(log_record.attributes or {}))

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        """nothing is buffered here.

        :param timeout_millis: unused
        :ptype timeout_millis: int
        :return: always ``True``
        :rtype: bool
        """
        return True

    def shutdown(self) -> None:
        """note the shutdown."""
        self.shut_down = True


def _export_through(exporter: FakeLogRecordExporter) -> OtelLogExport:
    return start_log_export(Resource.create({"service.name": "guard"}), "http://unused/otlp/v1/logs", exporter=exporter)


class TestTheLogsApiStillCarriesARecordToTheExporter:
    def test_a_python_log_record_reaches_the_exporter_through_the_handler(self) -> None:
        """the whole path setup.py relies on: handler, provider, batch processor, exporter."""
        exporter = FakeLogRecordExporter()
        export = _export_through(exporter)
        logger = logging.getLogger("threetears.observe.tests.otel_internals")
        logger.addHandler(export.handler)
        logger.setLevel(logging.INFO)
        try:
            logger.info("guard record")
            export.force_flush(timeout_millis=2000)
        finally:
            logger.removeHandler(export.handler)
            export.shutdown()

        assert exporter.bodies == ["guard record"]
        assert exporter.shut_down

    def test_the_handler_is_a_logging_handler(self) -> None:
        """setup.py wraps it in a logging.Handler and attaches it to the root logger."""
        export = _export_through(FakeLogRecordExporter())
        try:
            assert isinstance(export.handler, logging.Handler)
        finally:
            export.shutdown()

    def test_the_handler_is_the_instrumentation_packages_not_the_deprecated_sdk_one(self) -> None:
        """the SDK's ``LoggingHandler`` warns that it is deprecated; building the export must not."""
        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            export = _export_through(FakeLogRecordExporter())
        try:
            assert isinstance(export.handler, LoggingHandler)
        finally:
            export.shutdown()

    def test_a_record_carries_the_code_location_attributes(self) -> None:
        """the call-site enrichment rewrites pathname/funcName/lineno for these attributes to carry.

        The SDK handler always exported them; the replacement exports them only when asked, so a
        handler built without asking would drop every call site Loki shows.
        """
        exporter = FakeLogRecordExporter()
        export = _export_through(exporter)
        logger = logging.getLogger("threetears.observe.tests.otel_internals.code")
        logger.addHandler(export.handler)
        logger.setLevel(logging.INFO)
        try:
            logger.info("located record")
            export.force_flush(timeout_millis=2000)
        finally:
            logger.removeHandler(export.handler)
            export.shutdown()

        (attributes,) = exporter.attributes
        assert str(attributes["code.file.path"]).endswith("test_otel_internals.py")
        assert attributes["code.function.name"] == "test_a_record_carries_the_code_location_attributes"
        assert isinstance(attributes["code.line.number"], int)

    def test_the_default_exporter_is_built_without_a_connection(self) -> None:
        """production passes no exporter: the OTLP HTTP one is built from the endpoint, offline."""
        export = start_log_export(Resource.create({"service.name": "guard"}), "http://127.0.0.1:9/otlp/v1/logs")
        export.shutdown()
