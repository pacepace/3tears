# 3tears-observe

Structured logging, tracing, and OpenTelemetry setup for 3tears applications.

## Modules

- `threetears.observe.logging` -- Structured logging with context correlation and automatic call-site capture.
- `threetears.observe.tracing` -- `@traced` decorator (zero-cost without OpenTelemetry).
- `threetears.observe.metrics` -- `counter`/`histogram`/`gauge` get-or-create accessors plus the `@metered` decorator built on them, records prometheus metrics (zero-cost without prometheus_client).
- `threetears.observe.setup` -- OpenTelemetry SDK bootstrap (TracerProvider, LoggerProvider, OTLP exporters).

## Installation

```bash
pip install 3tears-observe

# With OpenTelemetry SDK support, traces over OTLP/gRPC:
pip install 3tears-observe[otel]

# The same over OTLP/HTTP, with no gRPC installed (about 10 MB against about 49):
pip install 3tears-observe[otel-http]
```

With `otel-http`, set `TelemetryConfig(protocol="http/protobuf", endpoint="http://collector:4318")`;
spans go to `<endpoint>/v1/traces`. `protocol` defaults to `"grpc"`, which needs the `otel` extra.
Log export is OTLP/HTTP under either extra.
