"""Tests for threetears.observe.setup."""

from __future__ import annotations

import logging
import socket
import threading
import time
from collections.abc import Iterator
from datetime import timedelta
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from opentelemetry.instrumentation.utils import suppress_instrumentation
from opentelemetry.proto.collector.logs.v1.logs_service_pb2 import ExportLogsServiceRequest

from threetears.observe.setup import TelemetryConfig, force_flush_telemetry, init_telemetry, reset_telemetry


class TestTelemetryConfig:
    """TelemetryConfig dataclass defaults and construction."""

    def test_defaults(self):
        config = TelemetryConfig()
        assert config.enabled is False
        assert config.endpoint == "http://localhost:4317"
        assert config.service_name == "threetears"
        assert config.service_version == "0.1.0"
        assert config.sample_rate == 1.0
        assert config.export_timeout_seconds == 10
        assert config.loki_endpoint is None
        assert len(config.suppressed_env_vars) > 0

    def test_custom_config(self):
        config = TelemetryConfig(
            enabled=True,
            endpoint="http://tempo:4317",
            service_name="myapp",
            service_version="2.0.0",
            sample_rate=0.5,
            loki_endpoint="loki:3100",
        )
        assert config.enabled is True
        assert config.service_name == "myapp"
        assert config.sample_rate == 0.5
        assert config.loki_endpoint == "loki:3100"

    def test_frozen(self):
        config = TelemetryConfig()
        import dataclasses

        with __import__("pytest").raises(dataclasses.FrozenInstanceError):
            config.enabled = True  # type: ignore[misc]


class TestInitDisabled:
    """init_telemetry when disabled."""

    def test_init_returns_false_when_disabled(self):
        from threetears.observe.setup import init_telemetry

        config = TelemetryConfig(enabled=False)
        assert init_telemetry(config) is False


class TestResetTelemetry:
    """reset_telemetry for test isolation."""

    def test_reset_is_safe_when_not_initialized(self):
        # Should not raise even when nothing was initialized
        reset_telemetry()


#: the paths of every export the accepting collector received, in order
_RECEIVED_EXPORTS: list[str] = []

#: the decoded OTLP request body of every export the accepting collector received, in order
_RECEIVED_REQUESTS: list[ExportLogsServiceRequest] = []


class _AcceptingCollector(BaseHTTPRequestHandler):
    """an OTLP/HTTP log collector that accepts every export, so shutdown's flush returns at once."""

    def do_POST(self) -> None:  # noqa: N802 -- the stdlib's handler naming
        """accept one export.

        :return: nothing
        :rtype: None
        """
        _RECEIVED_EXPORTS.append(self.path)
        _RECEIVED_REQUESTS.append(
            ExportLogsServiceRequest.FromString(self.rfile.read(int(self.headers.get("Content-Length", "0"))))
        )
        self.send_response(200)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002 -- the stdlib's signature
        """keep the collector's access log out of the test output.

        :param format: the stdlib's format string
        :ptype format: str
        :param args: its arguments
        :ptype args: object
        :return: nothing
        :rtype: None
        """


@pytest.fixture
def log_export_handler() -> Iterator[logging.Handler]:
    """the handler ``init_telemetry`` attaches to the root logger when log export is configured.

    Log export goes to a local collector that accepts everything, so nothing leaves the machine
    and shutdown's flush is immediate; the handler is the one the host's records actually reach,
    found as the root handler the init call added.

    :return: the handler log export installed
    :rtype: Iterator[logging.Handler]
    """
    collector = ThreadingHTTPServer(("127.0.0.1", 0), _AcceptingCollector)
    thread = threading.Thread(target=collector.serve_forever, daemon=True)
    thread.start()
    before = list(logging.root.handlers)
    config = TelemetryConfig(
        enabled=True,
        endpoint="http://127.0.0.1:9",
        loki_endpoint=f"127.0.0.1:{collector.server_address[1]}",
    )
    added: list[logging.Handler] = []
    try:
        assert init_telemetry(config) is True
        added = [handler for handler in logging.root.handlers if handler not in before]
        assert len(added) == 1, f"log export should attach exactly one root handler, attached {added}"
        yield added[0]
    finally:
        reset_telemetry()
        collector.shutdown()
        collector.server_close()
    assert added[0] not in logging.root.handlers, "shutdown must detach the log export handler"


def _record() -> logging.LogRecord:
    """a record as the stdlib makes it, before any call-site enrichment.

    :return: the record
    :rtype: logging.LogRecord
    """
    return logging.LogRecord(
        name="test",
        level=logging.INFO,
        pathname="/full/path/module.py",
        lineno=10,
        msg="hello",
        args=(),
        exc_info=None,
    )


class TestCallSiteEnrichingHandler:
    """The log export handler enriches records with ThreeTearsLogger's call site before OTel sees them."""

    def test_enriches_pathname(self, log_export_handler: logging.Handler):
        record = _record()
        record.call_site_file = "module.py"  # type: ignore[attr-defined]
        record.call_site_class = "MyClass"  # type: ignore[attr-defined]
        record.call_site_func = "handle"  # type: ignore[attr-defined]
        record.call_site_line = 42  # type: ignore[attr-defined]

        log_export_handler.emit(record)

        assert record.pathname == "module.py"
        assert record.funcName == "MyClass.handle"
        assert record.lineno == 42

    def test_enriches_without_class(self, log_export_handler: logging.Handler):
        record = _record()
        record.call_site_func = "my_function"  # type: ignore[attr-defined]

        log_export_handler.emit(record)

        assert record.funcName == "my_function"
        assert record.pathname == "/full/path/module.py"
        assert record.lineno == 10


class TestForceFlushTelemetry:
    """the public flush: what a test or a short-lived job calls to see its telemetry exported now."""

    def test_with_nothing_configured_it_flushes_nothing_and_succeeds(self) -> None:
        reset_telemetry()
        assert force_flush_telemetry(timeout=timedelta(seconds=1)) is True

    def test_a_buffered_log_record_reaches_the_collector_on_flush(self, log_export_handler: logging.Handler) -> None:
        """the log export batches; without a flush a record emitted now is exported seconds later."""
        _RECEIVED_EXPORTS.clear()
        log_export_handler.emit(_record())

        assert force_flush_telemetry(timeout=timedelta(seconds=5)) is True

        assert any(path.endswith("/otlp/v1/logs") for path in _RECEIVED_EXPORTS), _RECEIVED_EXPORTS

    @pytest.mark.timeout(60)
    def test_an_unreachable_collector_cannot_hold_the_flush_past_its_timeout(
        self, unreachable_log_export: None, caplog: pytest.LogCaptureFixture
    ) -> None:
        """the exporter retries an unreachable collector for longer than the caller allowed.

        with the root logger at INFO the exporter's own retry warnings are records too; the flush
        must still come back within its timeout, report that it did not finish, and say which
        signal held it.
        """
        logging.getLogger("threetears.observe.tests.app").warning("a record the collector will never take")

        started = time.monotonic()
        flushed = force_flush_telemetry(timeout=timedelta(seconds=1))
        elapsed = time.monotonic() - started

        assert flushed is False
        assert elapsed < 1.0 + _FLUSH_GRACE_SECONDS, f"the flush took {elapsed:.2f}s against a 1s timeout"
        unfinished = [
            getattr(record, "extra_data", {}).get("signal")
            for record in caplog.records
            if record.getMessage() == "telemetry flush did not finish within its timeout"
        ]
        assert unfinished == ["logs"], [record.getMessage() for record in caplog.records]


#: how far past its timeout the flush may return: starting a thread and writing one warning
_FLUSH_GRACE_SECONDS = 0.5


@pytest.fixture
def root_at_info() -> Iterator[None]:
    """the root logger at INFO, as a host application typically runs it, restored afterwards.

    at that level OpenTelemetry's own exporter and SDK warnings are emitted rather than dropped.

    :return: nothing
    :rtype: Iterator[None]
    """
    level = logging.root.level
    logging.root.setLevel(logging.INFO)
    try:
        yield
    finally:
        logging.root.setLevel(level)


def _unbound_local_port() -> int:
    """a local TCP port nothing listens on, so every connection to it is refused.

    :return: the port
    :rtype: int
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        port: int = probe.getsockname()[1]
    return port


@pytest.fixture
def unreachable_log_export(root_at_info: None) -> Iterator[None]:
    """log export configured against a collector that refuses every connection.

    :param root_at_info: the root logger at INFO for the test's duration
    :ptype root_at_info: None
    :return: nothing
    :rtype: Iterator[None]
    """
    config = TelemetryConfig(
        enabled=True,
        endpoint="http://127.0.0.1:9",
        loki_endpoint=f"127.0.0.1:{_unbound_local_port()}",
    )
    try:
        assert init_telemetry(config) is True
        yield
    finally:
        reset_telemetry()


def _exported_records() -> list[tuple[str, str]]:
    """every log record the accepting collector received, as (instrumentation scope, body).

    the log handler names each record's instrumentation scope after the python logger that
    emitted it.

    :return: the records, in the order received
    :rtype: list[tuple[str, str]]
    """
    return [
        (scope_logs.scope.name, log_record.body.string_value)
        for request in _RECEIVED_REQUESTS
        for resource_logs in request.resource_logs
        for scope_logs in resource_logs.scope_logs
        for log_record in scope_logs.log_records
    ]


class TestOpenTelemetrysOwnRecordsAreNeverExported:
    """an export failure must never produce exports: the records exporting produces stay local."""

    _OPENTELEMETRY_LOGGERS = (
        "opentelemetry",
        "opentelemetry.exporter.otlp.proto.http._log_exporter",
        "opentelemetry.exporter.otlp.proto.grpc.exporter",
        "opentelemetry.sdk._shared_internal",
        "opentelemetry.sdk._logs._internal",
    )

    def test_only_the_application_records_reach_the_collector(
        self, log_export_handler: logging.Handler, root_at_info: None, caplog: pytest.LogCaptureFixture
    ) -> None:
        _RECEIVED_REQUESTS.clear()
        for name in self._OPENTELEMETRY_LOGGERS:
            logging.getLogger(name).warning("opentelemetry's own record from %s", name)
        # how the SDK's batch processor wraps every call into its exporter: the HTTP client logs
        # each connection attempt from inside it
        with suppress_instrumentation():
            logging.getLogger("urllib3.connectionpool").warning("a record logged on the export's behalf")
        logging.getLogger("threetears.observe.tests.app").warning("an application record")
        logging.getLogger("opentelemetry_lookalike").warning("a record from a logger that only shares the prefix")

        assert force_flush_telemetry(timeout=timedelta(seconds=5)) is True

        exported = _exported_records()
        assert ("threetears.observe.tests.app", "an application record") in exported, exported
        assert ("opentelemetry_lookalike", "a record from a logger that only shares the prefix") in exported, exported
        leaked = [
            (scope, body)
            for scope, body in exported
            if scope == "opentelemetry" or scope.startswith("opentelemetry.") or scope == "urllib3.connectionpool"
        ]
        assert not leaked, leaked
        # still reported locally: every other handler on the root logger receives them
        local = {
            record.name
            for record in caplog.records
            if record.getMessage().startswith(("opentelemetry's own", "a record logged on the export's behalf"))
        }
        assert local == {*self._OPENTELEMETRY_LOGGERS, "urllib3.connectionpool"}
