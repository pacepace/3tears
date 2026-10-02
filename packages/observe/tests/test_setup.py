"""Tests for threetears.observe.setup."""

from __future__ import annotations

import logging
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from threetears.observe.setup import TelemetryConfig, init_telemetry, reset_telemetry


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


class _AcceptingCollector(BaseHTTPRequestHandler):
    """an OTLP/HTTP log collector that accepts every export, so shutdown's flush returns at once."""

    def do_POST(self) -> None:  # noqa: N802 -- the stdlib's handler naming
        """accept one export.

        :return: nothing
        :rtype: None
        """
        self.rfile.read(int(self.headers.get("Content-Length", "0")))
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
