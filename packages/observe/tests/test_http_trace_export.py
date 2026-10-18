"""Trace export over OTLP/HTTP, and telemetry with no gRPC installed.

``TelemetryConfig.protocol`` picks the trace exporter. ``"grpc"``, the default, is what every host
already runs and needs the ``otel`` extra, whose gRPC exporter brings grpcio. ``"http/protobuf"``
posts to ``<endpoint>/v1/traces``, as the OTLP exporter spec defines a base endpoint, and runs on
the ``otel-http`` extra, which has no gRPC in it. Log export was already OTLP/HTTP.
"""

from __future__ import annotations

import subprocess
import sys
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from opentelemetry import trace
from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import ExportTraceServiceRequest

from threetears.observe.setup import TelemetryConfig, init_telemetry, reset_telemetry, shutdown_telemetry

#: (path, decoded request) of every export the collector received, in order; the request is
#: decoded only for a trace export, since log export can be pointed at the same collector
_RECEIVED: list[tuple[str, ExportTraceServiceRequest | None]] = []


class _TraceCollector(BaseHTTPRequestHandler):
    """an OTLP/HTTP trace collector that accepts every export."""

    def do_POST(self) -> None:  # noqa: N802 -- the stdlib's handler naming
        """accept one export.

        :return: nothing
        :rtype: None
        """
        body = self.rfile.read(int(self.headers.get("Content-Length", "0")))
        decoded = ExportTraceServiceRequest.FromString(body) if self.path == "/v1/traces" else None
        _RECEIVED.append((self.path, decoded))
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
def collector_url() -> Iterator[str]:
    """a local OTLP/HTTP collector, as the base URL a host would configure.

    :return: ``http://127.0.0.1:<port>``
    :rtype: Iterator[str]
    """
    _RECEIVED.clear()
    collector = ThreadingHTTPServer(("127.0.0.1", 0), _TraceCollector)
    thread = threading.Thread(target=collector.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{collector.server_address[1]}"
    finally:
        reset_telemetry()
        collector.shutdown()
        collector.server_close()


def _exported_span_names() -> list[str]:
    """the names of every span the collector received.

    :return: span names, in arrival order
    :rtype: list[str]
    """
    return [
        span.name
        for _, request in _RECEIVED
        if request is not None
        for resource_spans in request.resource_spans
        for scope_spans in resource_spans.scope_spans
        for span in scope_spans.spans
    ]


def test_grpc_stays_the_default_protocol() -> None:
    assert TelemetryConfig().protocol == "grpc"


@pytest.mark.parametrize("suffix", ["", "/"])
def test_an_http_span_is_posted_to_the_traces_path_under_the_endpoint(collector_url: str, suffix: str) -> None:
    config = TelemetryConfig(enabled=True, endpoint=collector_url + suffix, protocol="http/protobuf")
    assert init_telemetry(config) is True

    with trace.get_tracer(__name__).start_as_current_span("over-http"):
        pass
    shutdown_telemetry()

    assert [path for path, _ in _RECEIVED] == ["/v1/traces"]
    assert _exported_span_names() == ["over-http"]


def test_an_unknown_protocol_is_refused() -> None:
    config = TelemetryConfig(enabled=True, protocol="http/json")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="http/json"):
        init_telemetry(config)


#: refuses gRPC, as an install of the ``otel-http`` extra has none
BLOCK_GRPC = """
import importlib.abc, sys
_REFUSED = ("grpc", "opentelemetry.exporter.otlp.proto.grpc")
class _NoGrpc(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if any(name == refused or name.startswith(refused + ".") for refused in _REFUSED):
            raise ModuleNotFoundError(f"No module named {name!r}", name=name)
        return None
sys.meta_path.insert(0, _NoGrpc())
"""


def _run(code: str) -> subprocess.CompletedProcess[str]:
    """run *code* in a fresh interpreter.

    :param code: the program
    :ptype code: str
    :return: the finished process
    :rtype: subprocess.CompletedProcess[str]
    """
    return subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=False, timeout=60)


def test_traces_and_logs_export_over_http_with_no_grpc_installed(collector_url: str) -> None:
    code = BLOCK_GRPC + (
        "import logging\n"
        "from opentelemetry import trace\n"
        "from threetears.observe.setup import TelemetryConfig, init_telemetry, shutdown_telemetry\n"
        f"url = {collector_url!r}\n"
        "assert init_telemetry(TelemetryConfig(\n"
        "    enabled=True, endpoint=url, protocol='http/protobuf', loki_endpoint=url.removeprefix('http://'),\n"
        "))\n"
        "with trace.get_tracer('t').start_as_current_span('no-grpc'):\n"
        "    logging.getLogger('t').warning('a record')\n"
        "shutdown_telemetry()\n"
        "loaded = sorted(n for n in sys.modules if any(n == r or n.startswith(r + '.') for r in _REFUSED))\n"
        "assert not loaded, loaded\n"
        "print('exported')"
    )
    result = _run(code)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "exported"
    assert _exported_span_names() == ["no-grpc"]
    assert "/otlp/v1/logs" in [path for path, _ in _RECEIVED]


def test_grpc_export_without_grpc_installed_names_the_extra() -> None:
    code = BLOCK_GRPC + (
        "from threetears.observe.setup import TelemetryConfig, init_telemetry\n"
        "try:\n"
        "    init_telemetry(TelemetryConfig(enabled=True, endpoint='http://127.0.0.1:9'))\n"
        "except ImportError as exc:\n"
        "    assert '3tears-observe[otel]' in str(exc), str(exc)\n"
        "else:\n"
        "    raise AssertionError('gRPC trace export started without gRPC')\n"
        "print('refused')"
    )
    result = _run(code)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "refused"
