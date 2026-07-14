# 3tears-registry

`threetears.registry` -- multi-pod tool routing. Registration, a NATS
KV-backed catalog, discovery, a load-balancing call proxy, heartbeat
monitoring, and pluggable routing strategies.

## Problem

When tools run across multiple pods, an agent needs to find a healthy pod
that serves a given tool, route the call there, and detect when a pod goes
dark -- without every consumer reimplementing service discovery and load
balancing on top of raw NATS.

## What it does

- A NATS KV-backed tool catalog with registration and discovery.
- A load-balancing `CallProxy` for routing calls across healthy pods.
- Heartbeat monitoring to detect and evict dead pods.
- Pluggable routing strategies.

## Design philosophy

Defense in depth on authorization: the registry denies unconditionally when
`user_id` is missing or the namespace row isn't yet visible, rather than
defaulting to allow when state is ambiguous -- explicitly catching
registration races instead of assuming they can't happen. Only the
production authorizer is wired in; there is no dual-enforcement window and
no back-compat aliases carried forward "just in case."

## When to adopt

Any multi-pod deployment of `agent-tools` or MCP tools where calls need to
be routed to a healthy pod rather than a fixed address.

## Composes with

- [`nats`](nats.md) -- the transport and KV catalog backend.
- [`agent-acl`](agent-acl.md) -- the RBAC authorizer backing the
  registry's defense-in-depth authorization.
- [`agent-tools`](agent-tools.md) -- a typical tool source routed through
  the registry. There is no code-level relationship with `mcp` today --
  don't assume the two are pre-wired.

## Install

```bash
pip install 3tears-registry
```
