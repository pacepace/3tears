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

# With OpenTelemetry SDK support:
pip install 3tears-observe[otel]
```
