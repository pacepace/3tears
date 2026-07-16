# 3tears-agent-workspace

`threetears.agent.workspace` -- workspace entities, sandbox, format
handlers, and namespace-routed L3 access.

## Problem

An agent that reads, writes, or manipulates files needs a sandboxed place
to do it, with format-aware handling and namespace isolation -- without
every app building its own ad hoc file-scratch-space logic per agent.

## What it does

- Workspace entities on top of `core`'s three-tier caching.
- A sandbox for agent file operations.
- Format handlers for the file types an agent works with.
- Namespace-routed L3 access.

## Design philosophy

Workspace deliberately does not define its own audit envelope or consumer.
It publishes through the shared `agent-audit` envelope instead, so a single
platform-side consumer owns audit persistence across every domain rather
than workspace maintaining its own audit path in parallel. This is the same
"one blessed path per concern" principle audit itself is built on.

## When to adopt

Any agent that reads or writes files as part of its work and needs that
scoped to a sandbox rather than the host filesystem directly.

## Composes with

- [`core`](core.md) -- the three-tier entity/collection base.
- [`agent-tools`](agent-tools.md) -- workspace's own file operations
  (read/write/etc.) are implemented as `TearsTool` subclasses.
- [`agent-audit`](agent-audit.md) -- publishes through the shared envelope.
- [`agent-acl`](agent-acl.md) -- namespace access control.

## Install

```bash
pip install 3tears-agent-workspace
```
