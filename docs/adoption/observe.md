# 3tears-observe

`threetears.observe` -- structured logging, tracing, and OpenTelemetry setup
for 3tears applications.

## Problem

Debugging a distributed system without correlated logs and traces means
grepping timestamps across pods and hoping. Every app re-solves "how do I
get structured, correlated logs and optional tracing" slightly differently,
and most solve it partially.

## What it does

- Structured logging through a `threetears` logger with a `NullHandler` --
  silent until the host opts in, following the standard-library convention.
- `set_context()` / `clear_context()` for `ContextVar`-backed correlation
  tags -- a fully generic `**kwargs` call with no hardcoded field names;
  typical callers pass short keys like `cid`, `conv`, `user`, `agent`,
  `customer`.
- A `@traced` decorator: near-zero-overhead passthrough with OpenTelemetry
  not installed; real spans once it is.
- ASGI correlation middleware and a standalone `configure_logging()` for
  simple scripts.

## Design philosophy

Nothing in `observe` activates itself. It follows the same "no implicit
connections; dependency injection only" principle as the rest of the
platform: `observe` produces structured log records and spans, but the host
decides where they go, at what level, and whether tracing is even
installed. This keeps `observe` usable in a library that itself has no
opinion on your logging infrastructure.

## When to adopt

Any 3tears app, from the start. There's no cost to including it -- tracing
is a no-op without OpenTelemetry installed, and logging is silent until you
attach a handler.

## Composes with

- Used by every other package in the platform for logging and tracing; it
  has no dependency on the rest of the platform itself.

## Install

```bash
pip install 3tears-observe
```
