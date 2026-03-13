# Observability

3tears instruments itself. Host applications configure the collectors.

## Logging

3tears follows the Python library logging convention: every module creates a
logger with `get_logger(__name__)` and attaches a `NullHandler`. No formatters,
no handlers, no level configuration. Output is silent until the host app opts in.

### Host integration

Route all 3tears logs through your app's logging by configuring the `threetears`
logger hierarchy:

```python
import logging

handler = logging.StreamHandler()
handler.setFormatter(your_formatter)

threetears_logger = logging.getLogger("threetears")
threetears_logger.addHandler(handler)
threetears_logger.setLevel(logging.DEBUG)
```

Every log message from every 3tears package (core, agent-memory, agent-tools)
flows through this single hierarchy. Your formatter controls the output format.

### Context propagation

3tears exposes three `contextvars` for correlation:

```python
from threetears.core.logging import correlation_id, session_id, conversation_id, set_context

# Set context at request entry (e.g. middleware)
set_context(correlation_id="req-abc", session_id="sess-123", conversation_id="conv-456")
```

If the host app uses `ContextFormatter` from 3tears (or reads these context vars
in its own formatter), log lines include `[cid:...] [sid:...] [conv:...]`.

### Standalone apps

For simple scripts or standalone apps that just want working log output:

```python
from threetears.core.logging import configure_logging

configure_logging("DEBUG")  # adds a StreamHandler with ContextFormatter
```

## Tracing (OpenTelemetry)

3tears uses the `@traced` decorator on all significant public entry points.
When OpenTelemetry is not installed, the decorator is a pure passthrough with
zero overhead (a single `bool` check per call). When OTel is installed and a
tracer provider is configured, every decorated method produces a span.

### What is traced

| Package | Entry points |
|---------|-------------|
| **core** | `BaseCollection.get`, `save_entity`, `reload_entity`, `delete`, `invalidate_cache` |
| **agent-memory** | `MemoryExtractor.extract`, `_extract_candidates`, `_resolve_actions`, `_execute_actions` |
| **agent-memory** | `MemoryRetriever.retrieve`, `retrieve_with_candidates` |
| **agent-tools** | `ToolExecutor.invoke_with_tools` |
| **agent-tools** | `ToolRouter.route` |
| **agent-tools** | `McpClient.list_tools`, `invoke_tool` |
| **agent-tools** | `parse_document` |

### Span attributes

Every span includes:

- `duration_ms` — wall-clock time
- `ctx.correlation_id`, `ctx.session_id`, `ctx.conversation_id` — from context vars (if set)

`BaseCollection` spans additionally include:

- `cache.table` — the collection's table name
- `cache.hit_tier` — `"L1+"` or `"miss"` (on `get` only)

On error, spans record `StatusCode.ERROR` and the exception.

### Host integration

The host app configures OTel as usual — 3tears picks it up automatically:

```python
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

provider = TracerProvider()
provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
trace.set_tracer_provider(provider)

# That's it. All 3tears @traced functions now emit spans.
```

### Optional arguments

The `@traced` decorator supports optional arguments:

```python
from threetears.core.tracing import traced

@traced()                              # default: span named after function
@traced(name="custom.span.name")       # custom span name
@traced(record_args=True)              # log function arguments as span attributes
@traced(record_result=True)            # log return type and collection length
```

Sensitive parameters (`password`, `token`, `secret`, `key`, `api_key`, etc.)
are automatically redacted when `record_args=True`.

## Design principle

**Library instruments, host configures.** 3tears never installs handlers,
formatters, exporters, or providers. It creates loggers with `NullHandler` and
decorates methods with `@traced`. The host application decides where logs go and
where traces are collected. This means:

- Zero noise if the host doesn't configure anything
- Zero overhead for tracing if `opentelemetry` isn't installed
- Full visibility if the host opts in — without 3tears knowing anything about
  the host's logging format, trace backend, or deployment topology
