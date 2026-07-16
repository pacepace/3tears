# 3tears-agent-acl

`threetears.agent.acl` -- unified RBAC evaluator and cache. Groups, roles,
assignments, an evaluation hot path, and an introspection trail. Pure
Python.

## Problem

Scattered per-resource authorization code -- one check in `namespace_grants`,
another in workspace checks, another as a tool fnmatch list -- means "can
this actor do this action" gets answered differently depending which code
path asks. Every scattered answer is its own bug surface, and one test suite
can never cover all of them at once.

## What it does

- Groups, roles, and role assignments as first-class entities.
- One evaluation hot path: "can actor do action on namespace."
- An introspection trail for explaining why a decision was made.
- Pure Python, no framework dependency.

## Design philosophy

The single source of truth for authorization decisions across the platform.
Running identical evaluation logic everywhere means the answer to "can actor
X do Y" is byte-identical no matter which package asks, and one test suite
covers every caller instead of one per resource type. It supersedes the
scattered per-resource ACL code paths this replaced.

## When to adopt

Any app with more than one place that needs to answer "is this actor
authorized," especially once a second consumer (a tool, an MCP server, a
workspace) needs the same answer the first one already computes.

## Composes with

- [`core`](core.md) -- the three-tier entity/collection base its groups,
  roles, and assignments are built on.
- [`agent-audit`](agent-audit.md) -- authorization decisions are typically
  audited through the shared envelope.
- [`mcp`](mcp.md) -- a typical authorizer backend for per-tool RBAC.
- [`agent-workspace`](agent-workspace.md) -- a typical consumer.

Note: `agent-tools` deliberately does *not* depend on this package -- it
accepts an ACL cache typed as `Any` instead, so it stays authorizer-agnostic.

## Install

```bash
pip install 3tears-agent-acl
```
