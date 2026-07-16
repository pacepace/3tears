# 3tears-mcp

`threetears.mcp` -- a Model Context Protocol framework. RBAC-gated
`McpServer`, `McpTool` plus `register_tool`, an auth-aware HTTP client, and
pluggable identity and authorizer protocols.

## Problem

Every product that exposes an MCP server ends up reimplementing stdio
transport, JWT auth, error mapping, and per-tool RBAC from scratch --
usually with subtly different (and subtly wrong) authorization behavior
each time.

## What it does

- `McpServer` and `McpTool` plus `register_tool` for declaring tools.
- Per-tool RBAC, default-deny, backed by `mcp_tool_grants` and epoch
  broadcast for cache reload.
- An auth-aware `PlatformHttpClient`.
- Pluggable identity and authorizer protocols so a host can supply its own
  auth backend.

## Design philosophy

A shared framework so per-product MCP servers compose it instead of
reimplementing transport and auth. Enforces strict stdio discipline -- no
module may write to stdout/stderr, since any stray byte confuses the MCP
client -- guarded by an AST enforcement test, not just a code-review
convention. RBAC is per-tool and default-deny, not a single server-wide
gate.

## When to adopt

Any app exposing tools to an LLM agent over MCP, especially where different
tools need different authorization rules.

## Composes with

- [`nats`](nats.md) -- transport for RBAC-grant epoch broadcast.
- [`agent-acl`](agent-acl.md) -- typical authorizer backend.
- [`epoch`](epoch.md) -- cache-reload coherence for tool grants.

There is no code-level relationship between this package and `registry` in
either direction today -- combine them yourself if you want multi-pod
routing on top of MCP tools; don't assume they're pre-wired.

## Install

```bash
pip install 3tears-mcp
```
