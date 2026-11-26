---
artifact: observability-strategy
version: 2
depends_on:
  - artifact: nonfunctional-requirements
  - artifact: product-brief
last_validated: null
---

# Observability Strategy

## Summary

3tears is a library family. It emits logs, traces and Prometheus metrics, and the host application decides where they go (`docs/observability.md`). With nothing configured, logs are silent and tracing and metrics are no-ops. Failures that a library handler absorbs are recorded through one helper, `contain()`, which logs and counts them (`docs/exceptions.md`). The OpenTelemetry floor the code needs is tested in CI (`scripts/test-otel-floor.sh`, `.github/workflows/ci.yml`). This repo runs no collector, dashboard or alert rule. Alert rules are the host's job (`docs/l2-first-operations.md`, "What the hub adds").

Terms used here:

| Term | Meaning |
|---|---|
| Host | The application that embeds 3tears packages and runs them as pods |
| Contained failure | An exception a handler absorbs on purpose and records via `contain()` |
| Floor | The oldest OpenTelemetry release the `otel` extra allows (`packages/observe/pyproject.toml`) |

## What You Get

| When | You can | Source |
|---|---|---|
| A broad handler absorbs an error | Query `contained_errors_total{site, exc_type}` for how often each function absorbs what, and read the WARNING with traceback and `contained_site` | `packages/observe/src/threetears/observe/containment.py` |
| A request misbehaves across pods | Find every log line and span for it by `cid` (and `sid`, `conv` when the host sets them) | `docs/observability.md`, "Context propagation" |
| A cache read is slow | Open the `BaseCollection.get` span and read `cache.hit_tier` (`L1+` or `miss`) and `duration_ms` | `docs/observability.md`, "Span attributes" |
| NATS loses data under an L2-first table | Read `collection_epoch_tables_behind`, `threetears_l2_first_refills_total` and the per-refill WARNING | `docs/l2-first-operations.md` |
| The epoch change log fills | Read `threetears_epoch_changes_fill_ratio` (warn at 0.75) | `docs/l2-first-operations.md` |
| A scheduled fire is lost or misrouted | Read `threetears_scheduled_jobs_failures_total`, `..._fires_unrecorded_total` and the `unrouted_kind` ERROR | `packages/scheduled-jobs/README.md` |
| A pod's probe fails | Call `/healthz/live` or `/healthz/ready` with `?format=json` to see which check failed | `packages/observe/src/threetears/observe/health.py` |

Each row is a runbook trigger. Runbooks that exist are listed in the operational spec.

## Architecture

```
3tears code --> logging (threetears.* hierarchy) --> host handlers / OTel log export
            --> @traced spans (OpenTelemetry API) --> host TracerProvider --> OTLP collector
            --> prometheus_client instruments     --> host /metrics scrape
```

- Principle: "Library instruments, host configures". The library never installs handlers, formatters, exporters or providers on its own (`docs/observability.md`, "Design principle").
- Opt-in bootstrap: `init_telemetry(TelemetryConfig)` wires a tracer provider and OTLP log export for hosts that want it. It suppresses ambient `OTEL_*` env vars. A disabled or unreachable collector leaves NoOp providers (`packages/observe/src/threetears/observe/setup.py`).
- Collectors, storage and dashboards are outside this repo.

## Signal Types

### Logging

| Aspect | Behaviour | Source |
|---|---|---|
| Loggers | `get_logger(__name__)` returns a `ThreeTearsLogger` with a `NullHandler`; silent until the host attaches handlers | `packages/observe/src/threetears/observe/logging.py` |
| Format | The host's formatter, or `configure_logging(level)` which installs `ContextFormatter` | `docs/observability.md` |
| Context tags | Each context entry renders as `[key:value]`; `extra_data` is appended as JSON | `logging.py` |
| Call site | File, class, function and line come from stack inspection, and the OTel log handler is rewritten to carry them | `logging.py`, `setup.py` |
| Contained failures | WARNING by default, ERROR or CRITICAL on request, with traceback | `docs/exceptions.md` |
| Periodic loops | A failed tick logs at WARNING with `exc_info` and the loop continues | `packages/observe/src/threetears/observe/periodic.py` |
| Retention | Not set by this repo | |

### Metrics

- Backend: `prometheus_client`, optional. Without it every instrument is a no-op (`packages/observe/src/threetears/observe/metrics.py`).
- Accessors: `counter()`, `gauge()`, `histogram()` and the `@metered` decorator.
- `HealthServer` serves `/metrics` only when the host passes a `metrics_provider`; otherwise it answers 404 (`health.py`).
- Metrics the packages define, as found in source or docs:

| Metric | Meaning | Source |
|---|---|---|
| `contained_errors_total{site, exc_type}` | Absorbed exceptions | `docs/exceptions.md` |
| `collection_epoch_tables_behind` | Tables a follower is behind on | `docs/l2-first-operations.md` |
| `collection_epoch_change_refused_total` | Change records refused | `packages/epoch/src` |
| `threetears_epoch_changes_fill_ratio` | Epoch change log fill | `docs/l2-first-operations.md` |
| `threetears_l2_first_l3_refused_total{table}` | L3 refused a committed write | `docs/l2-first-operations.md` |
| `threetears_l2_first_records_missing_total{table}` | Change records missing | `docs/l2-first-operations.md` |
| `threetears_l2_first_refills_total{table}` | L2 refills from L3 | `docs/l2-first-operations.md` |
| `threetears_scheduled_jobs_fires_total`, `..._failures_total`, `..._fires_unrecorded_total` | Fire accounting, cardinality-bounded | `packages/scheduled-jobs/src` |

- In-flight request gauge: `InflightRequestsGauge` (`packages/observe/src/threetears/observe/inflight.py`), described as scrapable by a KEDA Prometheus scaler (`health.py`).

### Tracing

- `@traced` wraps the public entry points of core (`BaseCollection.get`, `save_entity`, `reload_entity`, `delete`, `invalidate_cache`), agent-memory (extract, retrieve) and agent-tools (`ToolExecutor.invoke_with_tools`, `ToolRouter.route`, `McpClient`, `parse_document`) (`docs/observability.md`, "What is traced").
- Without OpenTelemetry installed the decorator is a passthrough with one `bool` check per call.
- Every span carries `duration_ms` and the `ctx.*` attributes; errors set `StatusCode.ERROR` and record the exception.
- `OTelMiddleware` adds a root SERVER span for WebSocket connections. HTTP root spans come from the host's FastAPI instrumentation (`packages/observe/src/threetears/observe/middleware.py`).

## Instrumentation Layers

| Layer | What 3tears does | Source |
|---|---|---|
| Automatic | Nothing installed on its own; OpenTelemetry and Prometheus are optional extras | `packages/observe/pyproject.toml` |
| Declarative | `@traced`, `@metered` markers on entry points | `tracing.py`, `metrics.py` |
| Contextual | `ctx.*` span attributes, `cache.table`, `cache.hit_tier`, `[key:value]` log tags | `docs/observability.md` |
| Manual | `contain()` / `contained()` at boundaries; `set_span_attribute`; custom `counter()` calls | `containment.py`, `tracing.py` |

## Correlation Context

| Scope | Key | How it propagates |
|---|---|---|
| Request | `cid` | `CorrelationMiddleware` reads or generates `X-Correlation-ID`, calls `set_context`, echoes the header; the header name and key are fixed |
| Session | `sid` | Host calls `set_context(sid=...)` |
| Domain | `conv` | Host calls `set_context(conv=...)` |

- The context is a `contextvars` dict. Keys are free-form; only `cid` is fixed by the middleware (`packages/observe/src/threetears/observe/middleware.py`, `logging.py`).
- Spans copy `cid`, `sid` and `conv` into `ctx.correlation_id`, `ctx.session_id` and `ctx.conversation_id` (`docs/observability.md`).
- The middleware needs the `asgi` extra and is not re-exported from `threetears.observe` (`middleware.py`).

## Sensitive Data Filtering

| Approach | Present | Source |
|---|---|---|
| Blocklist | `@traced(record_args=True)` skips parameters named `password`, `token`, `secret`, `key`, `api_key`, `encryption_key`, `jwt`, `auth`, `password_hash`; string attributes cut at 256 characters | `packages/observe/src/threetears/observe/tracing.py` |
| Default off | Arguments and results are not recorded unless the decorator asks | `docs/observability.md` |
| Erasure marker | `ANONYMIZED_MARKER` (`"[anonymized]"`) is the one value an erased field holds | `packages/observe/src/threetears/observe/erasure.py` |
| Log redaction | None found in `logging.py`; `extra_data` is dumped as given | `logging.py` |

The blocklist matches exact parameter names. Whether a log redactor is wanted is an open question.

## Alerting

This repo defines no alert rules. It names the signals a host should alert on:

- Any increase in `threetears_l2_first_l3_refused_total` or `threetears_l2_first_records_missing_total`.
- `threetears_epoch_changes_fill_ratio` at or above 0.75 (`EPOCH_CHANGES_FILL_WARN_RATIO`).
- Growth in `collection_epoch_tables_behind`.

Source: `docs/l2-first-operations.md`. The hub owns the rules and the thresholds' deployment.

## Health Signals

| Endpoint | Answers | Failure effect |
|---|---|---|
| `/healthz/live` | Every `LIVE` check passes (terminal NATS close, persistent auth wedge) | `503`; orchestrator restarts |
| `/healthz/ready`, `/healthz`, `/readyz` | Every `LIVE` and `READY` check passes | `503`; pulled from rotation |
| `?format=json` | Per-component `healthy` and `detail` | none |

Source: `packages/observe/src/threetears/observe/health.py`. Liveness is contained in readiness by design. A host can also run `nats.ping()` and `SELECT 1` (`docs/integration-guide.md`, Step 6).

## Infrastructure and Deployment

- Opt-in and zero cost when unconfigured: NullHandler logging, NoOp OpenTelemetry, no-op Prometheus (`docs/observability.md`, `metrics.py`).
- Extras: `otel` (gRPC) and `otel-http` (OTLP over HTTP, no gRPC), floors pinned `>=1.40` with a major cap (`packages/observe/pyproject.toml`, `setup.py`).
- The module `_otel_internals.py` is the one owner of OpenTelemetry's private names, because the logs API ships only as private modules (`packages/observe/src/threetears/observe/_otel_internals.py`).
- Flush and shutdown are bounded: `force_flush_telemetry(timeout=2s)`, `shutdown_telemetry()` (`setup.py`).

## Agent-Accessible Observability

Not applicable to this repo as a service. Hosts get the same surface through `HealthServer` JSON and `/metrics`. No MCP or CLI tool for querying logs, metrics or traces ships in `packages/observe`.

## Verification

| Check | How | Source |
|---|---|---|
| OTel works at the declared floor | CI step "Test observe at its declared OpenTelemetry floor" runs `scripts/test-otel-floor.sh`, which reads the `otel` extra, turns each `>=X` into `==X`, and runs the OTel tests in an isolated environment | `.github/workflows/ci.yml`, `scripts/test-otel-floor.sh` |
| The floor step stays in CI | Enforcement test | `tests/enforcement/test_otel_floor.py` |
| OTel private names still exist | Guard test | `packages/observe/tests/test_otel_internals.py` (named in `_otel_internals.py`) |
| No silent swallow | Enforcement test over every `src` tree | `tests/enforcement/test_no_silent_swallow.py` |
| Contained programming errors fail tests | Teardown tripwire for `TypeError`, `AttributeError`, `NameError`, `AssertionError` | `tests/contain_tripwire.py`, `tests/enforcement/test_contain_tripwire.py` |
| Middleware stays framework-free | `test_middleware_does_not_import_framework` | `middleware.py` |

## Direction

Ratified 2026-11-26 from declarations the owner already made; each Why cites where.

- **A failure either reaches the caller, or it is recorded where an operator will see it, never
  neither.** `except Exception` re-raises or calls `threetears.observe.contain(exc, log=...,
  what=...)`; `suppress(Exception)` is `with contained(...)`; `except CancelledError` re-raises,
  and another task is stopped with `await cancel_and_wait(task)`. An expected condition catches
  its specific type. A log line or a `# NOSILENT` comment does not excuse a broad handler.
  Why: a handler that logs and then returns `None`, `[]` or a partial result has still lied to
  its caller, and a soft-fail wrapper hides the failure from both caller and operator.
  `CLAUDE.md` § Exceptions; `docs/exceptions.md`.

## Candidate norms (unratified)

| Rule the code enforces | Enforcing file |
|---|---|
| A broad `except Exception` re-raises or calls `contain()`; `except BaseException` and `except CancelledError` re-raise | `tests/enforcement/test_no_silent_swallow.py` |
| A narrow handler whose body is only `pass`, `return None`, `continue` or `break` logs, re-raises or carries `# NOSILENT: <reason>` | `tests/enforcement/test_no_silent_swallow.py` |
| Stop another task with `cancel_and_wait`, not `cancel()` plus `await` | `docs/exceptions.md` (rule), `tests/enforcement/test_no_silent_swallow.py` |
| The OpenTelemetry floor step stays in CI | `tests/enforcement/test_otel_floor.py` |
| Only `_otel_internals.py` touches OpenTelemetry private names | `tests/enforcement/test_underscore_access.py` (exemption list `tests/enforcement/_underscore_exemptions.txt`) |
| A KV bucket grant matches the bucket the client opens | `tests/enforcement/test_kv_bucket_grant_naming.py` |

## Open questions

1. Is a log-side redactor wanted? `ContextFormatter` dumps `extra_data` unfiltered, and the blocklist covers only span arguments.
2. Which metrics are the supported public set? Names above come from docs and grep, with no single catalogue in the repo.
3. Retention, sampling defaults beyond `sample_rate=1.0`, and Loki use (`TelemetryConfig.loki_endpoint`) have no documented operator guidance.
