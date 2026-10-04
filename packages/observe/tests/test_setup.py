"""Tests for threetears.observe.setup."""

from __future__ import annotations

import json
import logging
import socket
import subprocess
import sys
import threading
import time
from collections.abc import Iterator
from datetime import timedelta
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from opentelemetry import trace
from opentelemetry.instrumentation.utils import suppress_instrumentation
from opentelemetry.proto.collector.logs.v1.logs_service_pb2 import ExportLogsServiceRequest
from opentelemetry.sdk.trace import SpanProcessor, TracerProvider

from threetears.observe.setup import (
    TelemetryConfig,
    force_flush_telemetry,
    init_telemetry,
    reset_telemetry,
    shutdown_telemetry,
)


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
        logging.getLogger("threetears.observe.tests.app").warning("an application record")
        logging.getLogger("opentelemetry_lookalike").warning("a record from a logger that only shares the prefix")

        assert force_flush_telemetry(timeout=timedelta(seconds=5)) is True

        exported = _exported_records()
        assert ("threetears.observe.tests.app", "an application record") in exported, exported
        assert ("opentelemetry_lookalike", "a record from a logger that only shares the prefix") in exported, exported
        leaked = [
            (scope, body) for scope, body in exported if scope == "opentelemetry" or scope.startswith("opentelemetry.")
        ]
        assert not leaked, leaked
        # still reported locally: every other handler on the root logger receives them
        local = {record.name for record in caplog.records if record.getMessage().startswith("opentelemetry's own")}
        assert local == set(self._OPENTELEMETRY_LOGGERS)

    def test_a_hosts_own_records_under_suppressed_instrumentation_are_still_exported(
        self, log_export_handler: logging.Handler, root_at_info: None
    ) -> None:
        """suppression marks an export only on the threads that export; a host's own block is its own business."""
        _RECEIVED_REQUESTS.clear()
        with suppress_instrumentation():
            logging.getLogger("threetears.observe.tests.app").warning("logged inside the host's suppressed block")

        assert force_flush_telemetry(timeout=timedelta(seconds=5)) is True

        exported = _exported_records()
        assert ("threetears.observe.tests.app", "logged inside the host's suppressed block") in exported, exported

    @pytest.mark.timeout(60)
    def test_what_the_http_client_logs_while_exporting_is_never_exported(
        self, log_export_handler: logging.Handler, root_at_debug: None, caplog: pytest.LogCaptureFixture
    ) -> None:
        """at DEBUG the HTTP client logs every request the exporter makes, on the exporting thread.

        exported, each export would queue the records of the next; two flushes in a row show it: the
        second exports whatever the first one's requests queued.
        """
        _RECEIVED_REQUESTS.clear()
        logging.getLogger("threetears.observe.tests.app").warning("an application record")

        assert force_flush_telemetry(timeout=timedelta(seconds=5)) is True
        assert force_flush_telemetry(timeout=timedelta(seconds=5)) is True

        on_the_exports_behalf = [record for record in caplog.records if record.name.startswith("urllib3.")]
        assert on_the_exports_behalf, "the HTTP client logged nothing while exporting; the test proves nothing"
        exported = _exported_records()
        assert ("threetears.observe.tests.app", "an application record") in exported, exported
        leaked = [(scope, body) for scope, body in exported if scope.startswith("urllib3.")]
        assert not leaked, leaked


@pytest.fixture
def root_at_debug() -> Iterator[None]:
    """the root logger and the HTTP client's logger at DEBUG, restored afterwards.

    the HTTP client's logger is set too: ``configure_logging`` quiets it to WARNING, and an earlier
    test in the same process may have called it.

    :return: nothing
    :rtype: Iterator[None]
    """
    http_client = logging.getLogger("urllib3")
    levels = (logging.root.level, http_client.level)
    logging.root.setLevel(logging.DEBUG)
    http_client.setLevel(logging.DEBUG)
    try:
        yield
    finally:
        logging.root.setLevel(levels[0])
        http_client.setLevel(levels[1])


class _BlockingSpanProcessor(SpanProcessor):
    """a span processor whose flush blocks until released: a provider stuck inside its exporter."""

    def __init__(self) -> None:
        """
        not yet entered, not yet released.
        """
        self.entered = threading.Event()
        self.release = threading.Event()

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        """block until released, ignoring the timeout as OpenTelemetry's batch processors do.

        :param timeout_millis: ignored
        :ptype timeout_millis: int
        :return: ``True`` once released
        :rtype: bool
        """
        self.entered.set()
        self.release.wait(30)
        return True


class _RaisingSpanProcessor(SpanProcessor):
    """a span processor whose flush raises, as a vendor exporter may."""

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        """raise.

        :param timeout_millis: ignored
        :ptype timeout_millis: int
        :return: never
        :rtype: bool
        :raises RuntimeError: always
        """
        raise RuntimeError("the flush exploded")


@pytest.fixture
def tracer_provider() -> Iterator[TracerProvider]:
    """tracing initialized with no log export; the SDK tracer provider ``init_telemetry`` installed.

    :return: the installed provider
    :rtype: Iterator[TracerProvider]
    """
    try:
        assert init_telemetry(TelemetryConfig(enabled=True, endpoint="http://127.0.0.1:9")) is True
        provider = trace.get_tracer_provider()
        assert isinstance(provider, TracerProvider)
        yield provider
    finally:
        reset_telemetry()


def _reports(caplog: pytest.LogCaptureFixture, message: str) -> list[dict[str, object]]:
    """the ``extra_data`` of every captured record with *message*.

    :param caplog: the captured records
    :ptype caplog: pytest.LogCaptureFixture
    :param message: the message to match exactly
    :ptype message: str
    :return: each matching record's extra data
    :rtype: list[dict[str, object]]
    """
    return [getattr(record, "extra_data", {}) for record in caplog.records if record.getMessage() == message]


class TestFlushFailuresAndConcurrency:
    """a flush that raises, a flush still running when another call arrives, and what each reports."""

    def test_a_flush_that_raises_is_reported_as_a_failure_and_the_later_ones_still_flush(
        self, log_export_handler: logging.Handler, caplog: pytest.LogCaptureFixture
    ) -> None:
        provider = trace.get_tracer_provider()
        assert isinstance(provider, TracerProvider)
        provider.add_span_processor(_RaisingSpanProcessor())
        _RECEIVED_REQUESTS.clear()
        logging.getLogger("threetears.observe.tests.app").warning("flushed after the traces failed")

        assert force_flush_telemetry(timeout=timedelta(seconds=5)) is False

        assert _reports(caplog, "telemetry flush failed") == [{"signal": "traces", "error_type": "RuntimeError"}]
        assert _reports(caplog, "telemetry flush did not finish within its timeout") == []
        exported = _exported_records()
        assert ("threetears.observe.tests.app", "flushed after the traces failed") in exported, exported

    @pytest.mark.timeout(60)
    def test_a_second_caller_waits_on_the_running_flush_instead_of_starting_another(
        self, tracer_provider: TracerProvider, caplog: pytest.LogCaptureFixture
    ) -> None:
        blocking = _BlockingSpanProcessor()
        tracer_provider.add_span_processor(blocking)
        first_result: list[bool] = []
        first = threading.Thread(
            target=lambda: first_result.append(force_flush_telemetry(timeout=timedelta(seconds=20))), daemon=True
        )
        first.start()
        try:
            assert blocking.entered.wait(5), "the first flush never reached the blocking provider"
            threads_before = set(threading.enumerate())
            started = time.monotonic()
            second = force_flush_telemetry(timeout=timedelta(seconds=0.5))
            elapsed = time.monotonic() - started
            started_threads = set(threading.enumerate()) - threads_before
        finally:
            blocking.release.set()
            first.join(10)

        assert second is False
        assert elapsed < 0.5 + _FLUSH_GRACE_SECONDS, f"the second flush took {elapsed:.2f}s against a 0.5s timeout"
        assert not started_threads, f"the second flush started {started_threads} beside the running one"
        held = _reports(caplog, "telemetry flush did not finish within its timeout")
        assert [report.get("signal") for report in held] == ["traces"], held
        assert held[0].get("held_by_an_earlier_flush") is True, held
        assert first_result == [True]

    @pytest.mark.timeout(120)
    def test_the_report_names_every_signal_not_started_when_the_first_one_holds_the_deadline(self) -> None:
        """traces, metrics and logs all configured; traces never finish, so neither of the others starts.

        a global meter provider can be installed only once per process, so this runs in its own.
        """
        completed = subprocess.run(
            [sys.executable, "-c", _NOT_STARTED_SCRIPT % _unbound_local_port()],
            capture_output=True,
            text=True,
            timeout=90,
            check=False,
        )
        assert completed.returncode == 0, completed.stderr
        outcome = json.loads(completed.stdout.strip().splitlines()[-1])
        assert outcome["flushed"] is False
        assert [(report["signal"], report["not_started"]) for report in outcome["reports"]] == [
            ("traces", ["metrics", "logs"])
        ], outcome


#: run in a fresh interpreter: block the traces flush with metrics and logs configured behind it
_NOT_STARTED_SCRIPT = """
import json
import logging
import os
import threading
from datetime import timedelta

from opentelemetry import metrics, trace
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.trace import SpanProcessor

from threetears.observe.setup import TelemetryConfig, force_flush_telemetry, init_telemetry

release = threading.Event()


class Blocking(SpanProcessor):
    def force_flush(self, timeout_millis=30000):
        release.wait(30)
        return True


reports = []


class Capture(logging.Handler):
    def emit(self, record):
        if record.getMessage() == "telemetry flush did not finish within its timeout":
            reports.append(record.extra_data)


metrics.set_meter_provider(MeterProvider())
logging.root.addHandler(Capture())
init_telemetry(TelemetryConfig(enabled=True, endpoint="http://127.0.0.1:9", loki_endpoint="127.0.0.1:%d"))
trace.get_tracer_provider().add_span_processor(Blocking())
flushed = force_flush_telemetry(timeout=timedelta(seconds=0.5))
outcome = {"flushed": flushed, "reports": [{"signal": r["signal"], "not_started": r["not_started"]} for r in reports]}
print(json.dumps(outcome), flush=True)
# the abandoned flush is still exporting to a port nothing listens on; the process is done with it
os._exit(0)
"""


#: how long the slow collector holds each export before accepting it: longer than shutdown's flush bound
_SLOW_COLLECTOR_SECONDS = 4.0


class _SlowCollector(BaseHTTPRequestHandler):
    """an OTLP/HTTP log collector that accepts every export, but only after holding it a while."""

    def do_POST(self) -> None:  # noqa: N802 -- the stdlib's handler naming
        """accept one export, late.

        :return: nothing
        :rtype: None
        """
        self.rfile.read(int(self.headers.get("Content-Length", "0")))
        time.sleep(_SLOW_COLLECTOR_SECONDS)
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
def slow_log_export() -> Iterator[None]:
    """log export configured against a collector that takes longer to accept than shutdown waits.

    :return: nothing
    :rtype: Iterator[None]
    """
    collector = ThreadingHTTPServer(("127.0.0.1", 0), _SlowCollector)
    collector.daemon_threads = True
    thread = threading.Thread(target=collector.serve_forever, daemon=True)
    thread.start()
    config = TelemetryConfig(
        enabled=True,
        endpoint="http://127.0.0.1:9",
        loki_endpoint=f"127.0.0.1:{collector.server_address[1]}",
    )
    try:
        assert init_telemetry(config) is True
        yield
    finally:
        reset_telemetry()
        collector.shutdown()
        collector.server_close()


class TestShutdownFlushIsBounded:
    """shutdown flushes each provider before shutting it down, and waits on that flush a bounded time."""

    @pytest.mark.timeout(60)
    def test_shutdown_stops_waiting_on_a_log_flush_after_its_bound(
        self, slow_log_export: None, caplog: pytest.LogCaptureFixture
    ) -> None:
        logging.getLogger("threetears.observe.tests.app").warning("a record the collector is slow to take")

        started = time.time()
        shutdown_telemetry()

        gave_up = [
            (record.created - started, getattr(record, "extra_data", {}).get("signal"))
            for record in caplog.records
            if record.getMessage() == "telemetry flush did not finish within its timeout"
        ]
        assert len(gave_up) == 1, gave_up
        waited, signal = gave_up[0]
        assert signal == "logs"
        assert waited < 2.0 + _FLUSH_GRACE_SECONDS, f"shutdown waited {waited:.2f}s on a flush bounded at 2s"
