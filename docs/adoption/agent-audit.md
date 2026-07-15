# 3tears-agent-audit

`threetears.agent.audit` -- one audit envelope and a fire-and-forget
`publish_audit` helper, with a single wire format and subject tree across
every domain.

## Problem

Domain-specific audit envelopes produce inconsistent wire shapes, which
forces UNION queries across tables just to answer "what happened" across
domains. And if publishing an audit event can fail the call that triggered
it, audit logging becomes a liability instead of a safety net.

## What it does

- One `AuditEvent` envelope, one wire format, used across every domain.
- `publish_audit()` -- durable-but-fire-and-forget: it persists the
  envelope to a durable NATS JetStream stream and awaits the broker's ack
  (at-least-once delivery), but a publish failure never raises back into
  the producing call.
- A single subject tree, so audit consumers subscribe once, not per domain.

## Design philosophy

Design commitments, stated explicitly: fire-and-forget, because audit is a
side effect and must never break the operation it's auditing; a typed wire
contract, to catch drift at construction time rather than at query time;
and no dual-emit -- when a package migrates onto this envelope, the legacy
envelope is deleted in the same change, not deprecated and left running in
parallel.

## When to adopt

Any package or app that needs to record "what happened" in a way that's
queryable consistently across domains, rather than per-feature ad hoc
logging.

## Composes with

- [`nats`](nats.md) -- the durable transport every audit event publishes
  through.
- [`agent-acl`](agent-acl.md) -- authorization decisions are typically
  audited through this envelope.
- [`agent-workspace`](agent-workspace.md) -- publishes through this shared
  envelope rather than defining its own.

## Install

```bash
pip install 3tears-agent-audit
```
